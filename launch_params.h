// This is a struct used to communicate launch parameters which are constant
// for all threads in a given optixLaunch call. 
struct Params
{
    uchar4* image;
    unsigned int  image_width;
    unsigned int  image_height;
    float3   cam_eye;
    float3   cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

// These structs represent the data blocks of our SBT records
struct RayGenData {
    // No data needed
};

struct HitGroupData {
    float3 albedo;
    float3* vertices;
    int3* indices;
};

struct MissData {
    float3 bg_color;
};
