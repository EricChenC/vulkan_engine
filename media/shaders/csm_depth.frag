#version 450

#extension GL_ARB_separate_shader_objects : enable

// layout (set = 1, binding = 0) uniform sampler2D colorMap;

// layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 color;

void main() 
{	
	// float alpha = texture(colorMap, inUV).a;
	// if (alpha < 0.5) {
		// discard;
	// }
    
    color = vec4(1.0, 1.0, 1.0, 1.0);
}