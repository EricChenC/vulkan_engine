#version 450
#extension GL_ARB_separate_shader_objects : enable

// matrices
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 lightPos;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 model;
} mt;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;

layout(location = 0) out vec4 viewNormal;
layout(location = 1) out vec4 viewLightDir;
layout(location = 2) out vec4 viewPos;



out gl_PerVertex {
    vec4 gl_Position;
};

// vertex shader
void main()
{
    gl_Position =  ubo.proj * ubo.view * mt.model * vec4(inPosition, 1.0);

    viewNormal = ubo.view * mt.model * vec4(inNormal, 0.0);
    
    viewLightDir =  ubo.view * vec4(ubo.lightPos, 0.0);
    
    viewPos = vec4(0.0, 0.0, 0.0, 0.0) - (ubo.view * mt.model * vec4(inPosition, 1.0));
    
}