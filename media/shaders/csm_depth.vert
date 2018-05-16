#version 450

#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec3 inPos;
// layout (location = 1) in vec2 inUV;

// todo: pass via specialization constant
#define SHADOW_MAP_CASCADE_COUNT 4

layout(push_constant) uniform ShadowPushConstBlock {
	uint cascadeIndex;
} spcb;

layout (binding = 0) uniform ShadowUniformBlock {
	mat4[SHADOW_MAP_CASCADE_COUNT] cascadeViewProjMat;
} sub;

// layout (location = 0) out vec2 outUV;

out gl_PerVertex {
	vec4 gl_Position;   
};

void main()
{
	// outUV = inUV;
	vec3 pos = inPos;
	gl_Position =  sub.cascadeViewProjMat[spcb.cascadeIndex] * vec4(pos, 1.0);
}