//
//  DetailViewController.swift
//  MonocularDepthEstimation
//
//  Created by Avishka Amunugama on 6/18/22.
//

import Cocoa

class DetailViewController: NSViewController {

    var onScreenChange: ((ScreenType) -> Void)?
    
    @IBOutlet weak var homeView: NSView!
    @IBOutlet weak var helpContainerView: NSView!
    @IBOutlet weak var depthEstimateContainerView: NSView!
    @IBOutlet weak var getStartedBtn: NSButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        applyTheme()
        homeView.isHidden = false
        helpContainerView.isHidden = true
        depthEstimateContainerView.isHidden = true
    }
    
    // Hides/Unhides the specified view
    func switchScreen(to screenType:ScreenType) {
        
        if homeView == nil ||
            helpContainerView == nil ||
            depthEstimateContainerView == nil {
            return
        }
        
        switch screenType {
        case .about:
            homeView.isHidden = false
            helpContainerView.isHidden = true
            depthEstimateContainerView.isHidden = true
        case .depth:
            homeView.isHidden = true
            helpContainerView.isHidden = true
            depthEstimateContainerView.isHidden = false
        case .help:
            homeView.isHidden = true
            helpContainerView.isHidden = false
            depthEstimateContainerView.isHidden = true
        }
    }
    
    // View help screen button action
    @IBAction func viewHelpScreen(_ sender: NSButton) {
        switchScreen(to: .help)
        if onScreenChange != nil {
            onScreenChange!(.help)
        }
    }
    
    // View depth estimation screen button action
    @IBAction func viewDepthEstimationScreen(_ sender: NSButton) {
        switchScreen(to: .depth)
        if onScreenChange != nil {
            onScreenChange!(.depth)
        }
    }
    
    // Basic UI customizations
    func applyTheme() {
        configureAsPrimaryButton(button: getStartedBtn, title: "Get Started")
    }
}
