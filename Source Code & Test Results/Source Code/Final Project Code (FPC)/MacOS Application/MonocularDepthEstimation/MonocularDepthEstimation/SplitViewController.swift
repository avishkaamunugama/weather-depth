//
//  SplitViewController.swift
//  WeatherDepth
//
//  Created by Avishka Amunugama on 6/28/22.
//

import Cocoa

class SplitViewController: NSSplitViewController {
    
    // Disables sidebar collapsing, sidebar will allways be visible.
    override func splitView(_ splitView: NSSplitView, canCollapseSubview subview: NSView) -> Bool {
        return false
    }
    
}
