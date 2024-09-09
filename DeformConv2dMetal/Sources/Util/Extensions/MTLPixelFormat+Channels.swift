import Foundation
import Metal

extension MTLPixelFormat {
    
    var channelsCount: Int? {
        switch self {
        case .r16Float, .r32Float:
            return 1
        case .rgba16Float, .rgba32Float:
            return 4
        default:
            return nil // unsupported
        }
    }
}
