//
//  OpenCVWrapper.h
//  MonocularDepthEstimation
//
//  Created by Avishka Amunugama on 6/12/22.
//

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject

+(NSImage *) cvtDepthMap:(CGImageRef)imageRef toStyle:(NSInteger)type;

@end

NS_ASSUME_NONNULL_END
