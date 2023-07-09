//
//  EstimateDepthViewController.swift
//  MonocularDepthEstimation
//
//  Created by Avishka Amunugama on 6/9/22.
//

import Cocoa
import Vision
import AppKit
import PythonKit


class EstimateDepthViewController: NSViewController {
    
    var model: VNCoreMLModel!
    var mapStyle: Int!
    var DEPTH_ESTIMATE: CGImage?
    
    @IBOutlet weak var srcImageView: DragNDropImageView!
    @IBOutlet weak var dstImageView: NSImageView!
    @IBOutlet weak var viewPointCloudBtn: NSButton!
    @IBOutlet weak var clearScreenBtn: NSButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        loadModel()
        applyTheme()
    }
    
    // Load CoreML Model
    func loadModel() {
        do {
            let model = try VNCoreMLModel(for: WeatherDepth(configuration: MLModelConfiguration()).model)
            self.model = model
        } catch {
            showAlertPopup(for: .failedToLoadModel)
            return
        }
    }
    
    /*
     Estimating a depth map for the input image
     */
    
    // Start depth estimation
    @IBAction func startProcess(_ sender: NSButton) {
        mapStyle = sender.tag
        predictDepth()
    }
    
    //Initiate prediction request
    func predictDepth() {
        let request  = VNCoreMLRequest(model: self.model, completionHandler: onVisionRequestComplete)
        request.imageCropAndScaleOption = .centerCrop
        
        guard let image = validInputImage() else { return }
        
        let trimedImage = image.trim(size: image.size)
        
        let handler = VNImageRequestHandler(ciImage: trimedImage.asCIImage()!, options: [:])
        try? handler.perform([request])
    }
    
    //On prediction request complete, converts the predictions to disparity and visualizes the depth map
    private func onVisionRequestComplete(request: VNRequest, error: Error?) {
        
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
              let depthMap = observations.first?.featureValue.multiArrayValue else { return }

        guard let cgImage = depthMap.cgImage(min: 0, max: 255, axes: (1,2,3)) else { return }
        guard let resizedCGImage = cgImage.resize(size: CGSize(width: 512, height: 512)) else { return }
        
        self.DEPTH_ESTIMATE = resizedCGImage // Saves the original predicted depth map
        
        guard let disparityMap = getDisparityMap(fromDepthMap: depthMap) else { return }
        guard let disparityImage = disparityMap.cgImage(min: 0, max: 255, axes: (1,2,3)) else { return }
        guard let resizedDisparityImage = disparityImage.resize(size: CGSize(width: 512, height: 512)) else { return }
        
        let nsImage = OpenCVWrapper.cvtDepthMap(resizedDisparityImage, toStyle: mapStyle)
        self.dstImageView.image = nsImage
    }
    
    // Converts the depth to disparity map for visualization
    // depth =  maxDepth/disparity
    private func getDisparityMap(fromDepthMap depth:MLMultiArray) -> MLMultiArray? {
        let number_of_channels = depth.shape[1].intValue
        let depth_w = depth.shape[2].intValue
        let depth_h = depth.shape[3].intValue

        if let disparityMap = try? MLMultiArray(shape: depth.shape, dataType: MLMultiArrayDataType.float32) {

            for k in 0..<number_of_channels {
                for i in 0..<depth_w {
                    for j in 0..<depth_h {
                        let index = k*(depth_w*depth_h) + i*(depth_h) + j
                        let disparity = NSNumber(value: 1000/depth[index].floatValue)
                        disparityMap[index] = disparity
                    }
                }
            }
            return disparityMap
        }
        return nil
    }
    
    /*
     Visualizing 3D Point Cloud using Open3D Python
     */
    
    @IBAction func viewPointCloud(_ sender: NSButton) {
        guard (validDepthMap() != nil) else { return }
        
        // Temporarily saves the depth maps
        let depthDestinationURL = getDocumentsDirectory().appendingPathComponent("temp_depth.jpg")
        let imageDestinationURL = getDocumentsDirectory().appendingPathComponent("temp.jpg")
        exportDepthMap(url: depthDestinationURL)
        exportTrimmedImage(url: imageDestinationURL)
        
        // Using the saved depth map for generating the point cloud
        let strDepthPath = depthDestinationURL.absoluteString.replacingOccurrences(of: "file://", with: "").replacingOccurrences(of: "%20", with: " ")
        let strImagePath = imageDestinationURL.absoluteString.replacingOccurrences(of: "file://", with: "").replacingOccurrences(of: "%20", with: " ")
        generatePointCloud(depthPath:strDepthPath, imagePath:strImagePath)
        removeTempDataFromDocumentsDirectory(tempData: [depthDestinationURL, imageDestinationURL])
    }
    
    // Generates a point cloud and visualizes it in a Tkinter window
    func generatePointCloud(depthPath:String, imagePath:String) {
        let dirPath = "/Users/avishka/Workspace/MacOS Projects/MonocularDepthEstimation/MonocularDepthEstimation/CustomComponents/"
        let sys = Python.import("sys")
        sys.path.append(dirPath)
        let piontCloud = Python.import("3DPointCloud")
        print("STARTED")
        piontCloud.createPointCloud(depthPath, imagePath)
        print("FINISHED")
    }
    
    // Temporary save path
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    
    // Remove all temporary data
    func removeTempDataFromDocumentsDirectory(tempData:[URL]){
        let fileManager = FileManager.default
        
        for url in tempData {
            try? fileManager.removeItem(at: url)
        }
    }
    
    /*
     Exporting Predicted Depth Map
     */
    
    // Exports results button action
    @IBAction func onExportResults(_ sender: Any) {
        guard (validInputImage() != nil) else { return }
        guard (validDepthMap() != nil) else { return }
        guard let srcUrl = self.srcImageView.url else { return }
        
        let panel = NSOpenPanel()
        panel.title = "Export Results"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = srcUrl.deletingLastPathComponent()
        
        panel.begin { (result) in
            if result.rawValue != NSApplication.ModalResponse.OK.rawValue { return }
            guard let url = panel.url else { return }
            let filename = srcUrl.deletingPathExtension().lastPathComponent
            let depthDestinationURL = url.appendingPathComponent("\(filename)_depth.jpg")
            let imageDestinationURL = url.appendingPathComponent("\(filename).jpg")
            
            self.exportDepthMap(url: depthDestinationURL)
            self.exportTrimmedImage(url: imageDestinationURL)
            showAlertPopup(for: .saveSuccess)
        }
    }
    
    // Exports grayscale depth map
    func exportDepthMap(url: URL) {
        guard let depthMap = validDepthMap() else { return }
        self.saveToJpg(cgImage: depthMap, url: url)
    }
    
    // Exports RBG trimmed input image
    func exportTrimmedImage(url: URL) {
        guard let image = validInputImage() else { return }

        let deviceScaleFactor = NSScreen.main?.backingScaleFactor ?? 2.0
        let trimedImage = image.trim(size: CGSize(width: 512/deviceScaleFactor, height: 512/deviceScaleFactor))

        guard let cgImage = trimedImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            showAlertPopup(for: .saveFailed)
            return
        }

        self.saveToJpg(cgImage: cgImage, url: url)
    }
    
    /*
     Clearing the screen
     */
    
    @IBAction func clearScreen(_ sender: NSButton) {
        self.srcImageView.image = NSImage(named: "dragNdrop")
        self.dstImageView.image = nil
        self.srcImageView.hasValidImage = false
        self.DEPTH_ESTIMATE = nil
    }
    
    /*
     Accessory Functions
     */
    
    // Save JPG image
    func saveToJpg(cgImage:CGImage, url: URL) {

        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        bitmapRep.size = CGSize(width: cgImage.width, height: cgImage.height)
        let jpegImgData = bitmapRep.representation(using: .jpeg, properties: [:])

        do {
            try jpegImgData?.write(to: url, options: .atomic)
        }
        catch {
            showAlertPopup(for: .saveFailed)
            return
        }
    }
    
    // Validates if prediction available
    func validDepthMap() -> CGImage? {
        guard (validInputImage() != nil) else { return nil }
        
        if let baseCGDepthImage = DEPTH_ESTIMATE {
            return baseCGDepthImage
        }
        else {
            showAlertPopup(for: .noDepthEstimate)
            return nil
        }
    }
    
    // Validates the input image
    func validInputImage() -> NSImage? {
        if let image = self.srcImageView.image {
            if self.srcImageView.hasValidImage {
                return image
            }
            else {
                showAlertPopup(for: .noInput)
                return nil
            }
        }
        else {
            showAlertPopup(for: .invalidInput)
            return nil
        }
    }
    
    // Basic UI customizations
    func applyTheme() {
        configureAsSecondaryButton(button: viewPointCloudBtn, title: "View 3D Point Cloud")
        configureAsTertiaryButton(button: clearScreenBtn, title: "Clear")
    }
}

extension CGImage {
    // Resize CGImage
    func resize(size:CGSize) -> CGImage? {
        let width: Int = Int(size.width)
        let height: Int = Int(size.height)
        
        let bytesPerPixel = self.bitsPerPixel / self.bitsPerComponent
        let destBytesPerRow = width * bytesPerPixel
        
        
        guard let colorSpace = self.colorSpace else { return nil }
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: self.bitsPerComponent, bytesPerRow: destBytesPerRow, space: colorSpace, bitmapInfo: self.alphaInfo.rawValue) else { return nil }
        
        context.interpolationQuality = .high
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return context.makeImage()
    }
}

extension NSImage {
    // Resize NSImage
    func resize(size: CGSize) -> NSImage {
        let dstRect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
        
        let result = NSImage(size: size)
        result.lockFocus()
        self.draw(in: dstRect)
        result.unlockFocus()
        return result
    }
    
    // Trim NSImage
    func trim(size: CGSize) -> NSImage {
        let targetAspect: CGFloat = 1.0
        let imageAspect = self.size.width / self.size.height
        
        let rect: CGRect
        if imageAspect == targetAspect {
            return self.resize(size: size)
        }
        else if targetAspect < imageAspect {
            let w = self.size.height * targetAspect
            rect = CGRect(x: (self.size.width - w) / 2, y: 0, width: w, height: self.size.height)
        } else {
            let h = size.width / targetAspect
            rect = CGRect(x: 0, y: (self.size.height - h) / 2, width: self.size.width, height: h)
        }
        
        let result = NSImage(size: size)
        result.lockFocus()
        
        let destRect = CGRect(origin: .zero, size: result.size)
        self.draw(in: destRect, from: rect, operation: .copy, fraction: 1.0)
        result.unlockFocus()
        return result
    }
    
    // Image format conversions
    
    func asCIImage() -> CIImage? {
       if let cgImage = self.asCGImage() {
          return CIImage(cgImage: cgImage)
       }
       return nil
    }
    
    func asCGImage() -> CGImage? {
       var rect = NSRect(origin: CGPoint(x: 0, y: 0), size: self.size)
       return self.cgImage(forProposedRect: &rect, context: NSGraphicsContext.current, hints: nil)
     }
}
