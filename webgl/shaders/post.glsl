//// fullscreen.vert
#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

//// bloom_down.frag
#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uTexel;      // 1 / source size
uniform float uKaris;     // 1 on the first level (firefly suppression)
uniform float uClampMax;  // pre-clamp of the source radiance
out vec4 o;
vec3 tap(vec2 uv) { return min(texture(uSrc, uv).rgb, vec3(uClampMax)); }
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
void main() {
  vec2 uv = gl_FragCoord.xy * 2.0 * uTexel;
  vec2 t = uTexel;
  // 13-tap downsample (jimenez 2014)
  vec3 a = tap(uv + t * vec2(-2, -2)), b = tap(uv + t * vec2(0, -2)), c = tap(uv + t * vec2(2, -2));
  vec3 d = tap(uv + t * vec2(-1, -1)), e = tap(uv + t * vec2(1, -1));
  vec3 f = tap(uv + t * vec2(-2, 0)), g = tap(uv), h = tap(uv + t * vec2(2, 0));
  vec3 i = tap(uv + t * vec2(-1, 1)), j = tap(uv + t * vec2(1, 1));
  vec3 k = tap(uv + t * vec2(-2, 2)), l = tap(uv + t * vec2(0, 2)), m = tap(uv + t * vec2(2, 2));
  vec3 g0 = (d + e + i + j) * 0.25, g1 = (a + b + f + g) * 0.25, g2 = (b + c + g + h) * 0.25;
  vec3 g3 = (f + g + k + l) * 0.25, g4 = (g + h + l + m) * 0.25;
  vec4 w = vec4(0.125);
  float w0 = 0.5;
  if (uKaris > 0.5) {
    w0 *= 1.0 / (1.0 + lum(g0));
    w = vec4(1.0 / (1.0 + lum(g1)), 1.0 / (1.0 + lum(g2)), 1.0 / (1.0 + lum(g3)), 1.0 / (1.0 + lum(g4))) * 0.125;
  }
  vec3 s = g0 * w0 + g1 * w.x + g2 * w.y + g3 * w.z + g4 * w.w;
  s /= (w0 + w.x + w.y + w.z + w.w);
  o = vec4(s, 1.0);
}

//// bloom_up.frag
#version 300 es
precision highp float;
uniform sampler2D uLow;   // smaller (already accumulated) level
uniform sampler2D uCur;   // this level's downsample
uniform vec2 uTexelLow;
uniform vec2 uSize;       // this level size
uniform float uRadius;
uniform float uMixCur;
uniform float uLowScale;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  vec2 t = uTexelLow * uRadius;
  vec3 s = texture(uLow, uv).rgb * 4.0;
  s += (texture(uLow, uv + vec2(-t.x, 0)).rgb + texture(uLow, uv + vec2(t.x, 0)).rgb +
        texture(uLow, uv + vec2(0, -t.y)).rgb + texture(uLow, uv + vec2(0, t.y)).rgb) * 2.0;
  s += texture(uLow, uv + vec2(-t.x, -t.y)).rgb + texture(uLow, uv + vec2(t.x, -t.y)).rgb +
       texture(uLow, uv + vec2(-t.x, t.y)).rgb + texture(uLow, uv + vec2(t.x, t.y)).rgb;
  s *= uLowScale / 16.0;
  o = vec4(s + texture(uCur, uv).rgb * uMixCur, 1.0);
}

//// composite.frag
#version 300 es
precision highp float;
uniform sampler2D uHDR;
uniform sampler2D uBloom;
uniform vec2 uSize;        // output size
uniform float uExposure;   // linear multiplier
uniform float uBloomAmt;
uniform float uVignette;
uniform float uSat;
uniform float uContrast;
uniform float uFrame;
uniform float uSatLin;    // saturation applied to linear hdr before the view transform
uniform float uBloomNorm;
uniform vec3 uTint;
uniform float uLook;      // 0 = ember (agx + look), 1 = neutral (khronos pbr neutral, matches the blender hero)
uniform float uRaw;       // 1: output linear coverage (shadow-mask validation view)
out vec4 o;

// agx (troy sobotka) — polynomial fit of the base contrast curve (b. wrensch)
vec3 agxContrast(vec3 x) {
  vec3 x2 = x * x, x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx(vec3 v) {
  const mat3 M = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                      0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  const float lo = -12.47393, hi = 4.026069;
  v = M * v;
  v = clamp(log2(max(v, vec3(1e-10))), lo, hi);
  v = (v - lo) / (hi - lo);
  return agxContrast(v);
}
vec3 agxLook(vec3 v) {
  float l = dot(v, vec3(0.2126, 0.7152, 0.0722));
  v = pow(max(v, 0.0), vec3(uContrast));
  return l + uSat * (v - l);
}
vec3 agxEotf(vec3 v) {
  const mat3 Mi = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  v = Mi * v;
  return pow(max(v, 0.0), vec3(2.2));
}
// khronos pbr neutral tone mapper (reference implementation, apache-2.0 spec)
vec3 pbrNeutral(vec3 color) {
  const float startCompression = 0.8 - 0.04;
  const float desaturation = 0.15;
  float x = min(color.r, min(color.g, color.b));
  float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  color -= offset;
  float peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) return color;
  const float d = 1.0 - startCompression;
  float newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;
  float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, vec3(newPeak), g);
}
vec3 srgbOETF(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  vec3 c = texture(uHDR, uv).rgb;
  if (uRaw > 0.5) { o = vec4(clamp(c, 0.0, 1.0), 1.0); return; }
  vec3 b = texture(uBloom, uv).rgb;
  c = mix(c, b / uBloomNorm, uBloomAmt);
  float cl = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = max(cl + uSatLin * (c - cl), vec3(0.0)) * uTint;
  c *= uExposure;
  // gentle vignette (cos^4-like)
  vec2 q = uv - 0.5;
  q.x *= uSize.x / uSize.y;
  float vig = 1.0 - uVignette * smoothstep(0.25, 1.05, length(q));
  c *= vig;
  vec3 m;
  if (uLook > 0.5) {
    // compress the ~7-stop doppler contrast (linear gamma 0.75), then pbr neutral
    m = pbrNeutral(pow(max(c, 0.0), vec3(0.75)));
    m = pow(max(m, 0.0), vec3(uContrast));
  } else {
    m = agxEotf(agxLook(agx(c)));
  }
  m = srgbOETF(clamp(m, 0.0, 1.0));
  // triangular dither (kills banding in the soft glow)
  float n = hash(gl_FragCoord.xy + fract(uFrame * 0.618) * 113.0) + hash(gl_FragCoord.xy * 1.37 + 7.0) - 1.0;
  m += n / 255.0;
  o = vec4(m, 1.0);
}

//// noise_bake.frag
#version 300 es
precision highp float;
uniform float uLayer;   // z texel index
uniform float uSize;    // texture size (e.g. 128)
uniform float uPeriod;  // lattice period (e.g. 16)
out vec4 o;
vec3 grad(vec3 c, float seed) {
  c = mod(c, uPeriod);
  vec3 p = fract(c * vec3(0.1031, 0.1030, 0.0973) + seed * 0.123);
  p += dot(p, p.yxz + 33.33);
  vec3 h = fract((p.xxy + p.yxx) * p.zyx) * 2.0 - 1.0;
  return normalize(h + 1e-4);
}
float gnoise(vec3 x, float seed) {
  vec3 i = floor(x), f = fract(x);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float n000 = dot(grad(i + vec3(0, 0, 0), seed), f - vec3(0, 0, 0));
  float n100 = dot(grad(i + vec3(1, 0, 0), seed), f - vec3(1, 0, 0));
  float n010 = dot(grad(i + vec3(0, 1, 0), seed), f - vec3(0, 1, 0));
  float n110 = dot(grad(i + vec3(1, 1, 0), seed), f - vec3(1, 1, 0));
  float n001 = dot(grad(i + vec3(0, 0, 1), seed), f - vec3(0, 0, 1));
  float n101 = dot(grad(i + vec3(1, 0, 1), seed), f - vec3(1, 0, 1));
  float n011 = dot(grad(i + vec3(0, 1, 1), seed), f - vec3(0, 1, 1));
  float n111 = dot(grad(i + vec3(1, 1, 1), seed), f - vec3(1, 1, 1));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
void main() {
  vec3 p = vec3(gl_FragCoord.xy, uLayer + 0.5) / uSize * uPeriod;
  float a = gnoise(p, 1.0) * 1.45;
  float b = gnoise(p + 31.7, 2.0) * 1.45;
  o = vec4(clamp(a * 0.5 + 0.5, 0.0, 1.0), clamp(b * 0.5 + 0.5, 0.0, 1.0), 0.0, 1.0);
}
