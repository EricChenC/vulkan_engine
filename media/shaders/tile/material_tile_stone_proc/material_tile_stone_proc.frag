#version 450
#extension GL_ARB_separate_shader_objects : enable

// data from vertex shader
layout(location = 1) in vec2 outTexcoord;
layout(location = 2) in vec3 outNormal;
layout(location = 3) in vec3 outLightPos;
layout(location = 4) in vec3 outFragPos;	// out view
layout(location = 5) in vec3 outPosition;

// normal 
layout(binding = 1) uniform UniformNormalParameters{
	vec4 ambientColor;						// ambient color
	vec4 diffuseColor;						// diffuse color
	vec4 specularColor;						// specular color
	vec4 transparency;						// transparency
	
	float diffuseRough;						// diffuse roughness
	float shininess;						// specular shininess
	float reflectivity;						// specular reflectivity
	float indexOfRefraction;				// index of refraction
	float extinction;						// extinction of metal
	float opacity;
	
	uint  options;
	uint  version;							// shader version
}unp;

// special
layout(binding = 6) uniform UniformSpecialParameters{
	vec4  tileColor;
	vec4  mortarColor;
	vec4  tileSpecularColor;

	float tileWidth;
	float tileHeight;
	float oddRowOffset;
	float mortarSize;
	float mortarThickness;
	float tileShininess;
	float tileReflectivity;
	float tileDiffRough;
	float tileStonyBumpFactor;
	float tileStonyBumpScale;
}usp;

// the environment map
//uniform sampler2D u_texReflection[7];

layout(location = 0) out vec4 outColor;

// Atom method
// gamma correction
vec4 gammaCorrection(float fGamma, vec4 col)
{
	vec4 vFactor = vec4(1.0 / fGamma);
	return pow(col, vFactor);
}

// Atom method
// a simple heck to smooth the transition of lit boundary
float fallOff(float a, float b, float x)
{
	if (x < a)  return 0.0;
	if (x >= b) return 1.0;
	x = (x - a) / (b - a);
	return x * x * (3.0 - 2.0 * x);
}

// Atom method
// for Schlick Fresnel computation
float pow5(float x)
{
	float foo = x * x;
	return foo * foo * x;
}

// Atom method
// Schlick Fresnel approximation
// Require pow5() method
float fresnelSchlick(float NdotV, float fReflectivity)
{
	if (fReflectivity >= 1.0) return 1.0;
	return fReflectivity + (1 - fReflectivity) * pow5(1.0 - NdotV);
}


// Atom method
// a better phong specular model
// the three vectors, V, L and N are viewing, lighting directions and surface normal
// sh: shininess in [0, 99]
float specularPhongAdv(vec3 V, vec3 L, vec3 N, float sh)
{
	vec3 R = normalize(reflect(V, N));
	float RdotL = max(dot(R, L), 0.0);
	vec3 specv = pow(vec3(RdotL), vec3(sh, sh * 2.0, sh * 8.0));// * vec3(0.333);
	return specv.x + specv.y + specv.z;
}

// Atom method
// a simple heck to simulate the Oren-Nayar effect in a cheap way
// fRoughness in [0, 1]. 0.0 means smooth surface like Lambert
// Require fallOff() heck to smooth shading boundary
float diffuseRough(float NdotL, float fRoughness)
{
	float fR = clamp (fRoughness, 0.0, 1.0);
	float foo = 1.0 - fR;
	return (foo + 1) * 0.5 * pow(NdotL, foo);
}

// Atom method
// unit vector to spherical coordinate and scaled to [0, 1] for
// reflection and background texture lookup
vec2 vector2ThetaPhi(vec3 v)
{
	float theta = atan(v.x, v.z);
	float phi = 3.1415926 * 0.5 - acos(v.y);
	return vec2((theta + 3.1415926) * (0.159155), 1.0 - 0.5 * (1.0 + sin(phi)));
}
vec2 hash22(vec2 p)
{ 
    float n = sin(dot(p, vec2(41, 289)));
    return fract(vec2(262144, 32768)*n); 
}

float Voronoi(vec2 p)
{	
    vec2 ip = floor(p); // Analogous to the cell's unique ID.
    p = fract(p); // Fractional reference point within the cell.

    float d = 10000.;

    for (float i = -1.; i < 1.1; i++)
    {
	    for (float j = -1.; j < 1.1; j++)
	    {
     	    vec2 cellRef = vec2(i, j); // Base cell reference point.
            vec2 offset = hash22(ip + cellRef); // 2D offset.
            // Vector from the point in the cell to the offset point.
            vec2 r = cellRef + offset - p; 
            float d2 = dot(r, r); // Squared length of the vector above.
            
            d = min(d, d2); // If it's less than the previous minimum, store it.
        }
    }

    return sqrt(d); 
}

vec3 stonyBump(vec2 uv)
{
    // layer number.
	float L;
	if(unp.version == 1){
		L = 6.;
	}else{
		L = 3.;
	}
	
    float sum = 0.001; // Amplitude sum, of sorts.
    
	vec3 val = vec3(0.);
    vec2 eps = vec2(0.01, 0.);
    
    // Infinite zoom loop.
	for (float i = 0.; i < L; i++)
	{	
        // Fractional time component. Obviously, incremented by "1./L" 
		float s = fract((i)/L);
        
		float e = exp2(s*L)*usp.tileStonyBumpFactor; // Range (approx): [ 1, pow(2., L)*gFreq ]
        float a = (1.-cos(s*6.283)) / e;  // Smooth transition.

        val.z += Voronoi(uv * e) * a; 		 // Sample value multiplied by the amplitude.
        val.x += Voronoi((uv-eps.xy)*e) * a; // nearby sample in the X-direction.
        val.y += Voronoi((uv-eps.yx)*e) * a; // nearby sample in the Y-direction.
        
        // Sum each amplitude. Used to normalize the results once the loop is complete.
        sum += a;
 	}
     
    // Normalizing the three Voronoi samples.
	val = val / sum;
    // Using the above to determine the dx and dy function gradients.
	val.xy = (val.xy - val.zz) / eps.x;
    // Using the gradient vector, "vec3(val.x, val.y, 0)," to perturb the XY plane normal ",vec3(0, 0, -1)."
    return normalize( vec3(0, 0, -1) + vec3(val.x, val.y, 0) * usp.tileStonyBumpScale * 0.1 );           
}

#define WIDTH_STEP		(usp.tileWidth + usp.mortarSize)
#define HEIGHT_STEP		(usp.tileHeight + usp.mortarSize)
#define BRICK_SIZE		vec2(usp.tileWidth, usp.tileHeight)
#define TILE_SIZE		vec2(WIDTH_STEP, HEIGHT_STEP)
#define MWF				(usp.mortarSize * 0.5 / WIDTH_STEP)
#define MHF				(usp.mortarSize * 0.5 / HEIGHT_STEP)

// Functions for procedure tiling
vec3 calcNormal(vec2 p)
{
	float hMortar = usp.mortarSize * 0.5;
	p = p * TILE_SIZE;
	vec3 vX = vec3(1.0, 0.0, 0.0);
	vec3 vY = vec3(0.0, 1.0, 0.0);
	
	float foox = p.x - usp.tileWidth - hMortar;
	float fooy = p.y - usp.tileHeight - hMortar;
	
	if (p.x < hMortar && p.x < p.y)
	{
	 	vX.z = 1.0;
	 	if (fooy > 0.0 && p.x > hMortar - fooy) vX.z = 0.0;
	}
	if (foox > 0.0 && hMortar - foox < p.y)
	{
		vX.z = -1.0;
		if (fooy > 0.0 && foox < fooy) vX.z = 0.0;
	}
	
	if (p.y < hMortar && p.y < p.x)
	{
		vY.z = 1.0;
		if (foox > 0.0 && p.y > hMortar - foox) vY.z = 0.0;
	}
	if (fooy > 0.0 && hMortar - fooy < p.x)
	{
		vY.z = -1.0;
		if (foox > 0.0 && fooy < foox) vY.z = 0.0;
	}

	return normalize(cross (vX, vY));
}

vec2 fragPos(vec2 coord)
{
	if (coord.x < 0.0) coord.x = coord.x + ceil(abs(coord.x));
	if (coord.y < 0.0) coord.y = coord.y + ceil(abs(coord.y));
	// This ranges from 0.0 to about 3.5. The integer component tells you the row number!
    float row = coord.y / usp.tileHeight;
    // calculate offset for odd row
    float fOffset = (fract(row * 0.5) > 0.5) ? usp.tileWidth * usp.oddRowOffset : 0.0;
	return vec2(fract((coord.x + fOffset) / usp.tileWidth), fract(coord.y / usp.tileHeight));
}



void main()
{
	// basic vectors
	vec3 vL = normalize(vec3(1.0));
	vec3 vV = normalize(outFragPos);
	vec3 vN = normalize(outNormal);
 
 	// falloff heck
	// This code should be place before the normal being modified by bump mapping
	// SNdorL: surface-normal-dot-lightdirection
	float SNdotL = max(dot(vN, vL), 0.0);
	float fFallOff = fallOff(0.0f, 0.25f, SNdotL);
  
	// procedure tiling
    vec2 ppos = fragPos(outTexcoord); // pigment position.
    vec2 rpos = ppos * TILE_SIZE;

	vec4 kd;
	vec3 vBumpNml;
	float hMortar = usp.mortarSize * 0.5;
	float bumpScale;
	if (rpos.x > hMortar && rpos.y > hMortar && rpos.x < hMortar + usp.tileWidth && rpos.y < hMortar + usp.tileHeight)
	{
		vBumpNml = stonyBump(outTexcoord);
		kd = vec4(vec3(usp.tileColor), 1.0);
	}
	else
	{
    	kd = vec4(vec3(usp.mortarColor), 1.0);// vec4(mix(vec3(usp.mortarColor), vec3(usp.tileColor), hori * vert), 1.0);
    	vBumpNml = calcNormal(ppos); // pigment normal.	
	}

	if(unp.version == 2){
		if (vBumpNml.z > 0.99)
		{
			kd = vec4(vec3(usp.tileColor), 1.0);
		}
	}
    
    // mapping to world coordinate
    if (dot(vN, vBumpNml) < 0.0) vBumpNml = -vBumpNml;
   	//vN = (normalize(v_tangent * vBumpNml.x + v_bitangent * vBumpNml.y + vN * vBumpNml.z)).xyz;
   	
   	vec4 ks = vec4(vec3(usp.tileSpecularColor), 1.0);

 	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	float reflFactor = fresnelSchlick(NdotV, usp.tileReflectivity);
	//float reflFactor = fresnelDielectric(NdotV, 1.f, iIoR);
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float tranFactor = 1.f - reflFactor;
	// do balancing
	ks = ks * reflFactor;
	kd = kd * tranFactor;

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for phong-like specular, which is more blur
	// specular reflection
	float shininess = usp.tileShininess;
	vec4 specRefl = vec4(0.0);
	float fT;
	if (shininess > 0)
	{
		int idx;
		vec3 vR = normalize(reflect(vV, vN));
		if (shininess <= 1.0)			// l = 6, 5
		{
			idx = 5;
			fT = 1.0 - shininess;
		}
		else if (shininess <= 4.0)		// l = 5, 4
		{
			idx = 4;
			fT = (4.0 - shininess) / 3.0; // 4 - 1 = 3
		}
		else if (shininess <= 16.0)		// l = 4, 3
		{
			idx = 3;
			fT = (16.0 - shininess) / 12.0; // 16 - 4 = 12
		}
		else if (shininess <= 48.0)		// l = 3, 2
		{
			idx = 2;
			fT = (48.0 - shininess) / 32.0; // 48 - 16 = 32
		}
		else if (shininess <= 100.0)	// l = 2, 1
		{
			idx = 1;
			fT = max((100 - shininess) / 100.0, 0.0);
		}
		
		//fT = pow(fT, 0.8);
		//vec4 spec0 = texture2D(u_texReflection[idx], vector2ThetaPhi(vR)) * (1.0 - fT);
		//vec4 spec1 = texture2D(u_texReflection[idx+1], vector2ThetaPhi(vR)) * fT;
		
		//specRefl = (spec1 + spec0) * ks;
	}

	// diffuse reflection (ambient)
	//vec4 diffRefl = kd * texture2D(u_texReflection[6], vector2ThetaPhi(vN));
	// total reflection contribution
	//vec4 cReflectinContribution = specRefl + diffRefl;
	//--- end of the reflection contribution computation
	
	float fSpecCoef = specularPhongAdv(vV, vL, vN, usp.tileShininess);
	float fDiffCoef = diffuseRough(max(dot(vN, vL), 0.0), usp.tileDiffRough);
   
	outColor = gammaCorrection(2.2, (ks * fSpecCoef + kd * fDiffCoef) * fFallOff);
	outColor.a = 1.0;
}