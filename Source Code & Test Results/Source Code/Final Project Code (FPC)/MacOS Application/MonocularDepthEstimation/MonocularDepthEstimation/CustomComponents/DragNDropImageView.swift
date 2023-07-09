//
//  DragNDropImageView.swift
//  MonocularDepthEstimation
//
//  Created by Avishka Amunugama on 6/9/22.
//

import Cocoa
import AppKit

class DragNDropImageView: NSImageView {
    
    var url: URL?
    var hasValidImage: Bool = false

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        return .copy
    }
    
    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        return .copy
    }
    
    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let pasteBoard = sender.draggingPasteboard
        
        guard let types = pasteBoard.types else { return true }
        
        if !types.contains(.fileURL) { return true }
        
        guard let url = NSURL.init(from: pasteBoard) as URL? else { return true }

        self.image = NSImage.init(contentsOf: url)
        self.url = url
        self.hasValidImage = true
        return true
    }
}
