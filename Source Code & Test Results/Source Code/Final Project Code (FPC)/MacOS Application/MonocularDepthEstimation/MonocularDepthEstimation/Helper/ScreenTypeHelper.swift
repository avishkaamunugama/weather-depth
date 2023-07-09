//
//  ScreenTypeHelper.swift
//  WeatherDepth
//
//  Created by Avishka Amunugama on 6/28/22.
//

import Foundation

enum ScreenType: Int, CaseIterable {
    case about = 0, depth = 1, help = 2
    
    var displayName: String {
        switch self {
        case .about:
            return "About"
        case .depth:
            return "Estimate Depth"
        case .help:
            return "Help"
        }
    }
}
