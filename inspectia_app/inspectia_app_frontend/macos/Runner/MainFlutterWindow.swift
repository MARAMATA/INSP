import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    let windowFrame = self.frame
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    // FORCER L'AFFICHAGE IMMÉDIAT
    self.center()
    self.makeKeyAndOrderFront(nil)   // IMPORTANT
    self.orderFrontRegardless()

    // Configuration pour permettre le redimensionnement
    self.isMovableByWindowBackground = true
    
    // Taille minimale de la fenêtre
    self.minSize = NSSize(width: 800, height: 600)
    
    // Configuration pour s'assurer que la fenêtre est visible
    self.level = .normal
    self.collectionBehavior = [.managed, .fullScreenPrimary]

    RegisterGeneratedPlugins(registry: flutterViewController)

    super.awakeFromNib()
  }
}
