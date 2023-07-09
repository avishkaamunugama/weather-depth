//
//  ThemeManager.swift
//  WeatherDepth
//
//  Created by Avishka Amunugama on 6/28/22.
//

import Foundation

func configureAsPrimaryButton(button: NSButton, title:String) {
    
    let pstyle = NSMutableParagraphStyle()
        pstyle.alignment = .center
    
    button.wantsLayer = true
    button.bezelStyle = .texturedSquare
    button.layer?.backgroundColor = NSColor.selectedContentBackgroundColor.cgColor
    button.layer?.cornerRadius = 8.0
    button.attributedTitle = NSAttributedString(string: title, attributes: [
        NSAttributedString.Key.foregroundColor : NSColor.white,
        NSAttributedString.Key.paragraphStyle : pstyle,
        NSAttributedString.Key.font : NSFont.systemFont(ofSize: 18, weight: .bold)
    ])
}

func configureAsSecondaryButton(button: NSButton, title:String) {
    
    let pstyle = NSMutableParagraphStyle()
        pstyle.alignment = .center
    
    button.wantsLayer = true
    button.bezelStyle = .texturedSquare
    button.layer?.backgroundColor = NSColor.selectedContentBackgroundColor.cgColor
    button.layer?.cornerRadius = 8.0
    button.attributedTitle = NSAttributedString(string: title, attributes: [
        NSAttributedString.Key.foregroundColor : NSColor.white,
        NSAttributedString.Key.paragraphStyle : pstyle,
        NSAttributedString.Key.font : NSFont.systemFont(ofSize: 14, weight: .medium)
    ])
}

func configureAsTertiaryButton(button: NSButton, title:String) {
    
    let pstyle = NSMutableParagraphStyle()
        pstyle.alignment = .center
    
    button.wantsLayer = true
    button.bezelStyle = .texturedSquare
    button.layer?.backgroundColor = NSColor.unemphasizedSelectedContentBackgroundColor.cgColor
    button.layer?.cornerRadius = 8.0
    button.attributedTitle = NSAttributedString(string: title, attributes: [
        NSAttributedString.Key.foregroundColor : NSColor.white,
        NSAttributedString.Key.paragraphStyle : pstyle,
        NSAttributedString.Key.font : NSFont.systemFont(ofSize: 16, weight: .medium)
    ])
}
