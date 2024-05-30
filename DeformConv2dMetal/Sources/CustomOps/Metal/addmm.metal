#include <metal_stdlib>
using namespace metal;

namespace dneprDroid {
    
    kernel void addmm(texture2d_array<float, access::read> m1 [[texture(0)]],
                      texture2d_array<float, access::read> m2 [[texture(1)]],
                      texture2d_array<float, access::write>  out [[texture(2)]],
                      uint3 gid [[thread_position_in_grid]]) {
        
        float4 sum = 0;
        const int m2_h = m2.get_height();
        
        for (int y2=0; y2< m2_h; y2++) {
            
            float4 v1 = m1.read(uint2(y2, 0), gid.z);
            float4 v2 = m2.read(uint2(gid.x, y2), gid.z);
            
            sum += v1 * v2;
        }
        out.write(sum, gid.xy, gid.z);
    }
    
}
