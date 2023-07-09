//
//  SideMenuViewController.swift
//  MonocularDepthEstimation
//
//  Created by Avishka Amunugama on 6/18/22.
//

import Cocoa


class SideMenuViewController: NSViewController {

    @IBOutlet weak var sideMenuTableView: NSTableView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        sideMenuTableView.delegate = self
        sideMenuTableView.dataSource = self
        sideMenuTableView.backgroundColor = .clear
        sideMenuTableView.selectRowIndexes(IndexSet(integer: ScreenType.about.rawValue), byExtendingSelection: false)
    }
}

// Sidebar tableview delegate and datasource functions
extension SideMenuViewController: NSTableViewDelegate, NSTableViewDataSource {
    
    // Number of rows
    func numberOfRows(in tableView: NSTableView) -> Int {
        return ScreenType.allCases.count
    }
    
    // Customizes cells
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        guard let cell = tableView.makeView(withIdentifier: tableColumn!.identifier, owner: self) as? NSTableCellView else { return nil }
        
        if let cell_lbl = cell.viewWithTag(102) as? NSTextField,
           let cell_img = cell.viewWithTag(101) as? NSImageView {
            cell_lbl.stringValue = ScreenType.allCases[row].displayName
            cell_img.image = NSImage(named: ScreenType.allCases[row].displayName)
        }
        return cell
    }
    
    // Handling tableview cell click actions
    func tableViewSelectionDidChange(_ notification: Notification) {
        guard sideMenuTableView.selectedRow != -1 else { return }
        guard let splitVC = parent as? NSSplitViewController else { return }
        if let detail = splitVC.children[1] as? DetailViewController {
            
            detail.switchScreen(to: ScreenType.allCases[sideMenuTableView.selectedRow])
            detail.onScreenChange = { screen in
                self.sideMenuTableView.selectRowIndexes(IndexSet(integer: screen.rawValue), byExtendingSelection: false)
            }
        }
    }
}
