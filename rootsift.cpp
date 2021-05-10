#include <opencv2/imgproc/imgproc.hpp>
#include "rootsift.hpp"

#include <iostream>
#include <chrono>
#include <pthread.h>



// number of threads to parallelize across
static const int N_THREADS = 1;

// SIFT parameters
static const int N_FEATURES = 0;
static const int N_OCTAVE_LAYERS = 3;
static const double CONTRAST_THRESHOLD = 0.04;
static const double EDGE_THRESHOLD = 10;
static const double SIGMA = 1.6;

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore cv::KeyPoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of cv::KeyPoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;


static void
unpackOctave( const cv::KeyPoint& kpt, int& octave, int& layer, float& scale )
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

static cv::Mat
createInitialImage( const cv::Mat& img, bool doubleImageSize, float sigma )
{
    cv::Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
    {
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, cv::DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
    }
    else
        img.convertTo(gray_fpt, cv::DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        cv::Mat dbl;
        resize(gray_fpt, dbl, cv::Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, cv::INTER_LINEAR);
        GaussianBlur(dbl, dbl, cv::Size(), sig_diff, sig_diff);
        return dbl;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        GaussianBlur(gray_fpt, gray_fpt, cv::Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}

static void
buildGaussianPyramid( const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves )
{
    std::vector<double> sig(N_OCTAVE_LAYERS + 3);
    pyr.resize(nOctaves*(N_OCTAVE_LAYERS + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = SIGMA;
    double k = pow( 2., 1. / N_OCTAVE_LAYERS );
    for( int i = 1; i < N_OCTAVE_LAYERS + 3; i++ )
    {
        double sig_prev = pow(k, (double)(i-1))*SIGMA;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < N_OCTAVE_LAYERS + 3; i++ )
        {
            cv::Mat& dst = pyr[o*(N_OCTAVE_LAYERS + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const cv::Mat& src = pyr[(o-1)*(N_OCTAVE_LAYERS + 3) + N_OCTAVE_LAYERS];
                resize(src, dst, cv::Size(src.cols/2, src.rows/2),
                       0, 0, cv::INTER_NEAREST);
            }
            else
            {
                const cv::Mat& src = pyr[o*(N_OCTAVE_LAYERS + 3) + i-1];
                GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }
}

static void
buildDoGPyramid( const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr )
{
    int nOctaves = (int)gpyr.size()/(N_OCTAVE_LAYERS + 3);
    dogpyr.resize( nOctaves*(N_OCTAVE_LAYERS + 2) );

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < N_OCTAVE_LAYERS + 2; i++ )
        {
            const cv::Mat& src1 = gpyr[o*(N_OCTAVE_LAYERS + 3) + i];
            const cv::Mat& src2 = gpyr[o*(N_OCTAVE_LAYERS + 3) + i + 1];
            cv::Mat& dst = dogpyr[o*(N_OCTAVE_LAYERS + 2) + i];
            subtract(src2, src1, dst, cv::noArray(), cv::DataType<sift_wt>::type);
        }
    }
}

static bool
adjustLocalExtrema( const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv,
                    int& layer, int& r, int& c, int nOctaveLayers,
                    float contrastThreshold, float edgeThreshold, float sigma )
{
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const cv::Mat& img = dog_pyr[idx];
        const cv::Mat& prev = dog_pyr[idx-1];
        const cv::Mat& next = dog_pyr[idx+1];

        cv::Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        cv::Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        cv::Vec3f X = H.solve(dD, cv::DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const cv::Mat& img = dog_pyr[idx];
        const cv::Mat& prev = dog_pyr[idx-1];
        const cv::Mat& next = dog_pyr[idx+1];
        cv::Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(cv::Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}

static float
calcOrientationHist( const cv::Mat& img, const cv::Point& pt, int radius,
                     float sigma, float* hist, int n )
{
    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    cv::AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::exp(W, W, len);
    cv::fastAtan2(Y, X, Ori, len, true);
    cv::magnitude(X, Y, Mag, len);

    for( k = 0; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];
    for( i = 0; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}

static void
findScaleSpaceExtrema( const std::vector<cv::Mat>& gauss_pyr,
                       const std::vector<cv::Mat>& dog_pyr,
                       std::vector<cv::KeyPoint>& keypoints )
{
    int nOctaves = (int)gauss_pyr.size()/(N_OCTAVE_LAYERS + 3);
    int threshold = cvFloor(0.5 * CONTRAST_THRESHOLD / N_OCTAVE_LAYERS * 255 * SIFT_FIXPT_SCALE);
    const int n = SIFT_ORI_HIST_BINS;
    float hist[n];
    cv::KeyPoint kpt;

    keypoints.clear();

    for( int o = 0; o < nOctaves; o++ )
        for( int i = 1; i <= N_OCTAVE_LAYERS; i++ )
        {
            int idx = o*(N_OCTAVE_LAYERS+2)+i;
            const cv::Mat& img = dog_pyr[idx];
            const cv::Mat& prev = dog_pyr[idx-1];
            const cv::Mat& next = dog_pyr[idx+1];
            int step = (int)img.step1();
            int rows = img.rows, cols = img.cols;

            for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
            {
                const sift_wt* currptr = img.ptr<sift_wt>(r);
                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
                const sift_wt* nextptr = next.ptr<sift_wt>(r);

                for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
                {
                    sift_wt val = currptr[c];

                    // find local extrema with pixel accuracy
                    if( std::abs(val) > threshold &&
                       ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
                         val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
                         val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
                         val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
                         val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
                         val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
                         val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
                         val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
                         val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
                        (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
                         val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
                         val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
                         val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
                         val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
                         val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
                         val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
                         val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
                         val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
                    {
                        int r1 = r, c1 = c, layer = i;
                        if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                                N_OCTAVE_LAYERS, (float)CONTRAST_THRESHOLD,
                                                (float)EDGE_THRESHOLD, (float)SIGMA) )
                            continue;
                        float scl_octv = kpt.size*0.5f/(1 << o);
                        float omax = calcOrientationHist(gauss_pyr[o*(N_OCTAVE_LAYERS+3) + layer],
                                                         cv::Point(c1, r1),
                                                         cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                         SIFT_ORI_SIG_FCTR * scl_octv,
                                                         hist, n);
                        float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                        for( int j = 0; j < n; j++ )
                        {
                            int l = j > 0 ? j - 1 : n - 1;
                            int r2 = j < n-1 ? j + 1 : 0;

                            if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                            {
                                float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                                kpt.angle = 360.f - (float)((360.f/n) * bin);
                                if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                    kpt.angle = 0.f;
                                keypoints.push_back(kpt);
                            }
                        }
                    }
                }
            }
        }
}

static int
descriptorSize()
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

static void
calcSIFTDescriptor( const cv::Mat& img, const cv::Point2f& ptf, float ori, float scl,
                    int d, int n, float* dst )
{
    cv::Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt((double) img.cols*img.cols + img.rows*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    cv::AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

    len = k;
    cv::fastAtan2(Y, X, Ori, len, true);
    cv::magnitude(X, Y, Mag, len);
    cv::exp(W, W, len);

    for( k = 0; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    for( k = 0; k < len; k++ )
        nrm2 += dst[k]*dst[k];
    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    for( i = 0, nrm2 = 0; i < k; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
    for( k = 0; k < len; k++ )
    {
        dst[k] = cv::saturate_cast<uchar>(dst[k]*nrm2);
    }
#else
    float nrm1 = 0;
    for( k = 0; k < len; k++ )
    {
        dst[k] *= nrm2;
        nrm1 += dst[k];
    }
    nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
    for( k = 0; k < len; k++ )
    {
        dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
    }
#endif
}

static void *
calcSIFTDescHelper( void *td )
{
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    struct thread_data *tdd = (struct thread_data *)td;

    for ( size_t i = tdd->thread_id; i < tdd->keypoints.size(); i += N_THREADS ) {
        cv::KeyPoint kpt = tdd->keypoints[i];
        int octave, layer;
        float scale;
        unpackOctave(kpt, octave, layer, scale);
        CV_Assert(octave >= tdd->firstOctave && layer <= tdd->nOctaveLayers+2);
        float size=kpt.size*scale;
        cv::Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
        const cv::Mat& img = tdd->gpyr[(octave - tdd->firstOctave)*(tdd->nOctaveLayers + 3) + layer];

        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;
        calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, tdd->descriptors.ptr<float>((int)i));

        // if (tdd->keypoints.size() - i < 15) {
        //     printf("thread %lu computed descriptor %lu\n", tdd->thread_id, i);
        // }
    }

    return nullptr;
}


static void
calcDescriptors( const std::vector<cv::Mat>& gpyr,
                 const std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    std::array<pthread_t, N_THREADS> threads;
    std::vector<struct thread_data> td(N_THREADS, {gpyr, keypoints, descriptors});

    // start threads
    for( size_t i = 0; i < N_THREADS; i++ )
    {
        td[i].nOctaveLayers = nOctaveLayers;
        td[i].firstOctave = firstOctave;
        td[i].thread_id = i;
        int rc = pthread_create(&threads[i], NULL, calcSIFTDescHelper, &td[i]);
        assert(!rc);
        printf("Created thread %lu\n", i);
    }

    // wait for threads to finish
    void *status;
    for ( size_t i = 0; i < N_THREADS; i++ )
    {
        int rc = pthread_join(threads[i], &status);
        assert(!rc);
    }
}

void
meSIFT( cv::InputArray& _image, cv::InputArray& _mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray _descriptors, bool useProvidedKeypoints )
{
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    cv::Mat image = _image.getMat(), mask = _mask.getMat();

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    if( useProvidedKeypoints )
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }

        firstOctave = std::min(firstOctave, 0);
        assert( firstOctave >= -1 && actualNLayers <= N_OCTAVE_LAYERS );
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    clock_t start = clock();

    cv::Mat base = createInitialImage(image, firstOctave < 0, (float)SIGMA);
    std::vector<cv::Mat> gpyr, dogpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log( (double)std::min( base.cols, base.rows ) ) / log(2.) - 2) - firstOctave;

    printf( "Time to init image and pyramids: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
    start = clock();

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    buildGaussianPyramid(base, gpyr, nOctaves);

    printf( "Time to build gaussian pyramid: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
    start = clock();

    buildDoGPyramid(gpyr, dogpyr);

    printf( "Time to build DoG pyramid: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
    start = clock();

    //t = (double)getTickCount() - t;
    //printf("pyramid construction time: %g\n", t*1000./tf);

    if( !useProvidedKeypoints ) // detect keypoints
    {
        //t = (double)getTickCount();
        findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
        cv::KeyPointsFilter::removeDuplicated( keypoints );

        if( N_FEATURES > 0 )
            cv::KeyPointsFilter::retainBest(keypoints, N_FEATURES);
        //t = (double)getTickCount() - t;
        //printf("cv::KeyPoint detection time: %g\n", t*1000./tf);

        if( firstOctave < 0 )
            for( size_t i = 0; i < keypoints.size(); i++ )
            {
                cv::KeyPoint& kpt = keypoints[i];
                float scale = 1.f/(float)(1 << -firstOctave);
                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                kpt.pt *= scale;
                kpt.size *= scale;
            }

        if( !mask.empty() )
            cv::KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }
    else
    {
        // filter cv::KeyPoints by mask
        //cv::KeyPointsFilter::runByPixelsMask( cv::KeyPoints, mask );
    }

    printf( "Time to detect keypoints: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
    start = clock();

    if( _descriptors.needed() ) // compute descriptors
    {
        //t = (double)getTickCount();
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, CV_32F);
        cv::Mat descriptors = _descriptors.getMat();

        calcDescriptors(gpyr, keypoints, descriptors, N_OCTAVE_LAYERS, firstOctave);
        //t = (double)getTickCount() - t;
        //printf("descriptor extraction time: %g\n", t*1000./tf);
    }

    printf( "Time to extract descriptors: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
}
