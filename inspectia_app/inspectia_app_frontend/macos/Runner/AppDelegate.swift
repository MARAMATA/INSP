import Cocoa
import FlutterMacOS

@main
class AppDelegate: FlutterAppDelegate {
  override func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    return true
  }

  override func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
    return true
  }
  
  override func applicationDidFinishLaunching(_ notification: Notification) {
    super.applicationDidFinishLaunching(notification)
    
    // Activer l'application de manière plus robuste
    DispatchQueue.main.async {
      NSApp.activate(ignoringOtherApps: true)
      
      // S'assurer que la fenêtre principale est visible
      if let window = NSApp.mainWindow {
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
      }
    }
  }
  
  override func applicationWillBecomeActive(_ notification: Notification) {
    super.applicationWillBecomeActive(notification)
    
    // S'assurer que l'application est au premier plan
    DispatchQueue.main.async {
      NSApp.activate(ignoringOtherApps: true)
      
      if let window = NSApp.mainWindow {
        window.makeKeyAndOrderFront(nil)
      }
    }
  }
  
  override func applicationDidBecomeActive(_ notification: Notification) {
    super.applicationDidBecomeActive(notification)
    
    // Activer l'application quand elle devient active
    DispatchQueue.main.async {
      NSApp.activate(ignoringOtherApps: true)
    }
  }
}
