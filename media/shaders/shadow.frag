#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec4 viewNormal;
layout(location = 1) in vec4 viewLightDir;
layout(location = 2) in vec4 viewPos;
layout(location = 3) in vec4 worldNormal;


layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outColor1;


#define pie 3.1415926

// main routine
void main()
{
    // Material properties
    vec3 diffuse = vec3(1.0, 1.0, 1.0); // or use texture
    vec3 ambient = vec3(0.4, 0.4, 0.4) * diffuse;
    vec3 specular = vec3(0.4, 0.4, 0.4);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    
    float lightPower = 1.2f;
    
    vec3 vl = normalize(viewLightDir.xyz);
    vec3 vn = normalize(viewNormal.xyz);
    vec3 vp = normalize(viewPos.xyz);
    
    float diffuseCos = clamp(dot(vn, vl), 0, 1);

    vec3 r = reflect(-vl, vn);
    
    float specularCos = clamp(dot(vp, r), 0, 1);
    
    float am = dot(vec4(0.4, 0.0, 0.0, 1.0), worldNormal);
    
    ambient *= am; // for fixed cube edge can't see problem
    
    // lambertian
    // lightColor * (diffuse / pie) * (lightPower * diffuseCos)
    
    vec3 color = ambient
    + lightColor * lightPower * diffuse * diffuseCos
    + lightColor * lightPower * specular * pow(specularCos, 10);
    
    outColor = vec4(color, 1.0);

	outColor1 = vec4(color.g, color.r, color.b, 1.0);
    
 }