//
//  CustomAlertHelper.swift
//  WeatherDepth
//
//  Created by Avishka Amunugama on 6/28/22.
//

import Foundation

enum CustomAlertType: String {
    case saveSuccess = "Successfully Export"
    case saveFailed = "Export Failed"
    case noInput = "No Input"
    case invalidInput = "Invalid Input"
    case noDepthEstimate = "No Depth Estimate"
    case failedToLoadModel = "Failed To Load Model"
    
    var message:String {
        switch self {
        case .saveSuccess:
            return "The depth map and the image were successfully saved to the chosen directory."
        case .saveFailed:
            return "Sorry the export operation didn't go as to plan. Please try again."
        case .noInput:
            return "Please add an image to estimate depth for."
        case .invalidInput:
            return "Sorry invalid input type. Please add an valid image of type PNG or JPG."
        case .noDepthEstimate:
            return "This function requires a depth map. Please add an input image and click predict to generate a depth map."
        case .failedToLoadModel:
            return "Oops! The depth estimation model did not load as expected. Please quit and reopen the application."
        }
    }
}

func showAlertPopup(for type:CustomAlertType) {
    let alert = NSAlert()
    alert.messageText = type.rawValue
    alert.informativeText = type.message
    alert.addButton(withTitle: "OK")
    alert.runModal()
}
