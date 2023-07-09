//
//  OpenCVWrapper.m
//  MonocularDepthEstimation
//
//  Created by Avishka Amunugama on 6/12/22.
//

#import "OpenCVWrapper.h"
#import <opencv2/opencv.hpp>

// Converts CGImage to Mat
static void CGImageToMat(CGImage *image, cv::Mat &mat) {

    // Create a pixel buffer.
    NSInteger width = CGImageGetWidth(image);
    NSInteger height = CGImageGetHeight(image);
    CGImageRef imageRef = image;
    
    cv::Mat mat8uc4 = cv::Mat((int)height, (int)width, CV_8UC4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef contextRef = CGBitmapContextCreate(mat8uc4.data, mat8uc4.cols, mat8uc4.rows, 8, mat8uc4.step, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);

    // Draw all pixels to the buffer.
    cv::Mat mat8uc3 = cv::Mat((int)width, (int)height, CV_8UC3);
    cv::cvtColor(mat8uc4, mat8uc3, cv::COLOR_RGBA2BGR);

    mat = mat8uc3;
}

// Converts Mat to NSImage
static NSImage *MatToNSImage(cv::Mat &mat) {

    // Create a pixel buffer.
    assert(mat.elemSize() == 1 || mat.elemSize() == 3);
    cv::Mat matrgb;
    if (mat.elemSize() == 1) {
        cv::cvtColor(mat, matrgb, cv::COLOR_GRAY2RGB);
    } else if (mat.elemSize() == 3) {
        cv::cvtColor(mat, matrgb, cv::COLOR_BGR2RGB);
    }

    // Change a image format.
    NSData *data = [NSData dataWithBytes:matrgb.data length:(matrgb.elemSize() * matrgb.total())];
    CGColorSpaceRef colorSpace;
    if (matrgb.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(matrgb.cols, matrgb.rows, 8, 8 * matrgb.elemSize(), matrgb.step.p[0], colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    NSBitmapImageRep *bitmapImageRep = [[NSBitmapImageRep alloc] initWithCGImage:imageRef];
    NSImage *image = [NSImage new];
    [image addRepresentation:bitmapImageRep];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return image;
}

// Returns a colored depth maps according to the selected color map stype
static NSImage *cvtImage(CGImageRef imageRef, cv::ColormapTypes type) {
    cv::Mat mat;
    CGImageToMat(imageRef, mat);
    
    cv::Mat coloredMat;
    applyColorMap(mat, coloredMat, type);
    
    NSImage *infernoImage = MatToNSImage(coloredMat);
    return infernoImage;
}

// Returns the original depth map as it is . Grayscale
static NSImage *cvtImage(CGImageRef imageRef) {
    cv::Mat mat;
    CGImageToMat(imageRef, mat);

    NSImage *infernoImage = MatToNSImage(mat);
    return infernoImage;
}


@implementation OpenCVWrapper

+ (NSImage*) cvtDepthMap:(CGImageRef)imageRef toStyle:(NSInteger)type {
    
    switch(type) {
        case 1:
            return cvtImage(imageRef, cv::COLORMAP_VIRIDIS);
        case 2:
            return cvtImage(imageRef, cv::COLORMAP_INFERNO);
        case 3:
            return cvtImage(imageRef, cv::COLORMAP_PLASMA);
        default:
            return cvtImage(imageRef);
    }
}


@end
