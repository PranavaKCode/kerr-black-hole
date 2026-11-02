#version 300 es
// real-time kerr null-geodesic tracer. units G = c = M = 1, bl hamiltonian with E = −p_t = 1:
//   2ΣH = Δ p_r² − P²/Δ + p_θ² + (L/sinθ − a sinθ)²,   P = r² + a² − aL
// integrated in mino time (dλ = Σ dσ) with the angular part pole-free: unit vector n,
// J = n × p_ang (J_z = L, Q = |J|² − L² − a² cos²θ)
//   dr/dσ   = Δ p_r
//   dp_r/dσ = −(r−1) p_r² + 2rP/Δ − (r−1) P²/Δ²
//   dn/dσ   = J × n + W ẑ × n            W = a (P/Δ − 1)  (frame dragging in bl φ)
//   dJ/dσ   = a² n_z (n × ẑ) + W ẑ × J
// rk4 with an adaptive step (checked against reference.py). the disk is volumetric emission and
// absorption, seen as a blackbody at g·T; stars are analytic, filtered with the lensing jacobian.
precision highp float;
precision highp int;
precision highp sampler3D;

out vec4 outColor;

uniform vec2  uRes;
uniform vec2  uJitter;      // sub-pixel jitter (pixels)
uniform vec2  uShift;       // lens shift (tan units)
uniform float uTanHalf;     // tan(vertical fov / 2)
uniform float uSeed;
uniform int   uMode;        // 0 beauty, 1 shadow mask, 2 redshift map, 3 ray fate, 4 probe (one pixel -> numbers)
uniform vec2  uProbe;       // probe pixel (render-target coordinates, mode 4)

// probe record: g, cylindrical radius and image order of the first strong disk contribution
vec4 gProbe = vec4(0.0, 0.0, 0.0, -1.0);

// spacetime + camera (precomputed on the cpu)
uniform float uA;
uniform float uRh;          // outer horizon
uniform float uRcap;        // capture radius (just outside r+)
uniform vec2  uIsco;        // E, L of the isco orbit (plunging flow)
uniform float uRisco;
uniform float uRo;          // observer radius
uniform float uThO;         // observer polar angle
uniform float uPhO;         // observer azimuth (rotates sky + disk pattern)
uniform vec3  uCam;         // sqrt(Σ/Δ), sqrt(Σ), sqrt(g_φφ)   at the camera
uniform vec2  uZamo;        // lapse α, zamo angular velocity ω
uniform float uRfar;

// disk
uniform float uRin, uRout, uThick, uTurb, uShear, uTime, uDensity, uDust, uPlunge, uTPeak, uNTmax;
uniform float uTGamma;      // visible-palette remap of the nt temperature profile (1 = nt shape)
uniform float uAnimBlend;   // 0 static (single noise phase), >0 two-phase crossfade (animating)
uniform float uDiskOn;

// relativity
uniform float uDoppler;     // 0 = film (no doppler / beaming), 1 = physical
uniform float uGrav;        // gravitational redshift (incl. time dilation) on/off
uniform float uRing;        // photon-ring emphasis (artistic boost of n>=2 images)

// render
uniform int   uSteps;
uniform float uStepK;
uniform float uDiskStep;
uniform int   uOctaves;     // turbulence octaves (quality)
uniform float uStars;
uniform float uGalaxy;
uniform float uLumNorm;     // 1 / Y(T_peak)
uniform float uPixAng;      // unlensed pixel angle (rad)

uniform sampler2D uBB;      // blackbody lut: rgb chroma, a = log2 luminance
uniform sampler3D uNoise;   // tileable baked gradient noise (rg)
uniform sampler2D uPrev;    // previous accumulation
uniform float uBlend;       // weight of this frame

const float PI = 3.14159265358979;

// utilities
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
vec3 hash33(vec3 p) {
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.xxy + p.yxx) * p.zyx);
}
float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

vec3 blackbody(float T) {
  // lut covers 800 K .. 80 000 K logarithmically
  if (T < 800.0) return vec3(0.0);
  float u = (log(T) - log(800.0)) / (log(80000.0) - log(800.0));
  vec4 t = textureLod(uBB, vec2(clamp(u, 0.0, 1.0), 0.5), 0.0);
  return t.rgb * exp2(t.a);
}
vec3 bbChroma(float T) {
  float u = (log(T) - log(800.0)) / (log(80000.0) - log(800.0));
  return textureLod(uBB, vec2(clamp(u, 0.0, 1.0), 0.5), 0.0).rgb;
}

// geodesic right-hand side
struct S { float r; float pr; vec3 n; vec3 J; };

S deriv(S y, float L) {
  float a = uA;
  float D = y.r * y.r - 2.0 * y.r + a * a;
  float iD = 1.0 / D;
  float P = y.r * y.r + a * a - a * L;
  float W = a * (P * iD - 1.0);
  S d;
  d.r = D * y.pr;
  d.pr = -(y.r - 1.0) * y.pr * y.pr + (2.0 * y.r * P - (y.r - 1.0) * P * P * iD) * iD;
  d.n = cross(y.J, y.n) + W * vec3(-y.n.y, y.n.x, 0.0);
  d.J = a * a * y.n.z * vec3(y.n.y, -y.n.x, 0.0) + W * vec3(-y.J.y, y.J.x, 0.0);
  return d;
}
S axpy(S y, S d, float h) {
  S o;
  o.r = y.r + h * d.r; o.pr = y.pr + h * d.pr; o.n = y.n + h * d.n; o.J = y.J + h * d.J;
  return o;
}

// disk model
float noiseA(vec3 p) { return textureLod(uNoise, p, 0.0).r * 2.0 - 1.0; }
float noiseB(vec3 p) { return textureLod(uNoise, p, 0.0).g * 2.0 - 1.0; }

const mat3 ROT = mat3( 0.00,  0.80,  0.60,
                      -0.80,  0.36, -0.48,
                      -0.60, -0.48,  0.64);

// turbulence field in co-rotating sheared coordinates -> (log-density perturbation, filament)
vec3 turbulence(vec3 q, float rc, float laneAng) {
  // eddy scale follows the disk thickness ~ r: shrink coordinates with radius (smooth warp)
  float sc = 0.075 * pow(max(rc, 1.0), -0.35);
  // fbm in the r channel; ridged filaments from the g channel of octaves 2 and 3 (shared lookups)
  vec3 p1 = q * sc;
  vec3 p2 = ROT * p1 * 2.07 + 0.137;
  vec3 p3 = ROT * p2 * 2.07 + 0.137;
  vec2 n1 = textureLod(uNoise, p1, 0.0).rg * 2.0 - 1.0;
  vec2 n2 = textureLod(uNoise, p2, 0.0).rg * 2.0 - 1.0;
  vec2 n3 = textureLod(uNoise, p3, 0.0).rg * 2.0 - 1.0;
  float f = 0.46 * n1.x + 0.285 * n2.x + 0.177 * n3.x;
  if (uOctaves >= 4) {
    vec3 p4 = ROT * p3 * 2.07 + 0.137;
    f += 0.110 * noiseA(p4);
    if (uOctaves >= 5) f += 0.068 * noiseA(ROT * p4 * 2.07 + 0.137);
  }
  float rg = 1.0 - abs(n2.y);
  rg = rg * rg * rg;
  float rg2 = 1.0 - abs(n3.y);
  rg2 = rg2 * rg2 * rg2 * rg2;
  // low-frequency ridges -> long, sheared dark dust lanes (spiral arcs, not rings);
  // extra winding for the lanes: older structures are more sheared
  float cl = cos(laneAng), sl = sin(laneAng);
  vec3 ql = vec3(cl * q.x - sl * q.y, sl * q.x + cl * q.y, q.z);
  vec2 l1 = textureLod(uNoise, ql * sc * 0.42 + vec3(5.3, 1.7, 9.1), 0.0).rg * 2.0 - 1.0;
  float lf = 1.0 - abs(l1.x + 0.55 * l1.y);   // two independent fields: irregular spacing
  lf *= lf * lf;
  // very low-frequency mask: lanes cluster in some sectors and vanish in others (less repetitive)
  lf *= smoothstep(-0.45, 0.2, noiseB(ql * sc * 0.11 + vec3(2.1, 7.7, 3.3)));
  return vec3(f, 0.65 * rg + 0.35 * rg2, lf);
}

float omegaK(float rc) { return 1.0 / (pow(rc, 1.5) + uA); }

// scale height of the (mildly flared) disk
float diskH(float rc) {
  float x = max(rc, uRisco);
  return uThick * x * pow(x / uRisco, 0.1);
}

// novikov–thorne-like temperature (normalised to 1 at its maximum)
float tempProfile(float rc) {
  float x = max(rc, uRin);
  float f = pow(x, -0.75) * pow(max(1.0 - 0.92 * sqrt(uRin / x), 0.0), 0.25);
  return f / uNTmax;
}

// radial density envelope
float radialEnv(float rc) {
  // tenuous (radiation-pressure dominated) rim at the isco, full density from ~1.8 r_in outwards
  float inner = smoothstep(uRin * 0.985, uRin * 1.10, rc) * mix(0.12, 1.0, smoothstep(uRin, uRin * 1.9, rc));
  float outer = 1.0 - smoothstep(uRout * 0.55, uRout * 1.05, rc);
  return inner * outer * pow(uRin / max(rc, uRin), 0.35);
}

// returns redshift factor g (obs/emit), plus gD (doppler part) and gG (gravitational part)
vec3 redshift(float r, float mu, float pr, float L, float rc) {
  float a = uA;
  float s2 = max(1.0 - mu * mu, 1e-6);
  float Sig = r * r + a * a * mu * mu;
  float D = r * r - 2.0 * r + a * a;
  if (rc >= uRisco) {
    float Om = omegaK(rc);
    float gtt = -(1.0 - 2.0 * r / Sig);
    float gtp = -2.0 * a * r * s2 / Sig;
    float gpp = (r * r + a * a + 2.0 * a * a * r * s2 / Sig) * s2;
    float nrm = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
    float ut = inversesqrt(max(nrm, 1e-5));
    float gD = 1.0 / max(1.0 - Om * L, 1e-3);
    float gG = 1.0 / ut;
    return vec3(gD * gG, gD, gG);
  } else {
    // plunging gas: conserved E, L of the isco orbit
    float E = uIsco.x, Lp = uIsco.y;
    float A = (r * r + a * a) * (r * r + a * a) - a * a * D * s2;
    float iSD = 1.0 / (Sig * D);
    float gTT = -A * iSD, gTP = -2.0 * a * r * iSD, gPP = (D - a * a * s2) * iSD / s2;
    float ut = -gTT * E + gTP * Lp;
    float up = -gTP * E + gPP * Lp;
    float q = gTT * E * E - 2.0 * gTP * E * Lp + gPP * Lp * Lp;
    float ur = -sqrt(max(-1.0 - q, 0.0) * D / Sig);
    float Eem = ut - L * up - pr * ur;
    float g = 1.0 / max(Eem, 1e-3);
    float gG = 1.0 / max(ut, 1e-3);
    return vec3(g, g / gG, gG);
  }
}

// one radiative-transfer segment of affine length dl at the sample point
void diskSample(float r, vec3 n, float pr, float L, float dl, int order,
                inout vec3 I, inout float Tr) {
  float a = uA;
  float s = length(n.xy);
  float R2 = sqrt(r * r + a * a);
  float rc = R2 * s;                  // cylindrical (bl) radius
  float z = r * n.z;
  float H = diskH(rc);
  float vz = z / H;
  if (vz * vz > 13.0) return;
  float plunge = rc < uRin ? 1.0 : 0.0;
  float env;
  if (plunge > 0.5) {
    env = uPlunge * exp(-(uRin - rc) / (0.18 * uRin)) * smoothstep(uRh * 1.02, uRh * 1.25, rc);
  } else {
    env = radialEnv(rc);
  }
  if (env < 1e-4) return;
  // cheap early-out: even the densest turbulent filament (~x16) would be negligible here
  if (env * exp(-0.5 * vz * vz) * uDensity * dl < 2e-4) return;

  // sheared, co-rotating coordinates (φ' = φ − Ω(r)·τ), camera azimuth applied to the pattern
  float Om = omegaK(max(rc, uRisco * 0.9));
  vec2 xy = n.xy / max(s, 1e-6) * rc;
  float ang0 = uPhO - Om * uShear;
  vec3 tb;
  {
    float ang = ang0 - Om * uTime;
    float c = cos(ang), sn = sin(ang);
    vec3 q = vec3(c * xy.x - sn * xy.y, sn * xy.x + c * xy.y, z * 2.2);
    float laneAng = -Om * uShear * 1.6;
    tb = turbulence(q, rc, laneAng);
    if (uAnimBlend > 0.0) {
      // second phase, half a period later, crossfaded so shear stays bounded while animating
      float ang2 = ang0 - Om * (uTime - uAnimBlend);
      float c2 = cos(ang2), s2 = sin(ang2);
      vec3 q2 = vec3(c2 * xy.x - s2 * xy.y, s2 * xy.x + c2 * xy.y, z * 2.2) + 17.3;
      vec3 tb2 = turbulence(q2, rc, laneAng);
      float w = 0.5 + 0.5 * cos(2.0 * PI * uTime / max(uAnimBlend * 2.0, 1e-3));
      tb = mix(tb2, tb, w);
    }
  }
  // vertical profile with turbulent surface
  float vert = exp(-0.5 * vz * vz * (1.0 - 0.18 * uTurb * tb.x));
  float logd = uTurb * (2.2 * tb.x + 2.1 * (tb.y - 0.3));
  float dens = env * vert * exp(logd - 0.5 * uTurb * uTurb);

  // dust lanes / gaps at a few radii (wobbling with the turbulence)
  float lane = 0.0;
  if (uDust > 0.0 && plunge < 0.5) {
    // dark lanes follow the sheared ridges of a low-frequency field; none in the hot inner disk
    lane = smoothstep(0.62, 0.93, tb.z) * smoothstep(uRin * 2.2, uRin * 5.0, rc);
    lane = clamp(lane * uDust * 1.2, 0.0, 0.95);
  }
  float T = uTPeak * pow(tempProfile(rc), uTGamma) * (1.0 + 0.16 * tb.x + 0.10 * tb.y);
  if (plunge > 0.5) T = uTPeak * 0.85 * pow(rc / uRin, 1.5);
  T *= (1.0 - 0.6 * lane);

  vec3 gg = redshift(r, n.z, pr, L, rc);
  float gD = pow(gg.y, uDoppler);
  float gG = mix(1.0, gg.z, uGrav);
  float g = gD * gG;

  float kap = uDensity * dens * (1.0 - 0.75 * lane) + uDensity * 0.6 * lane * env * vert;
  float dtau = kap * dl / max(g, 0.05);
  float boost = (order >= 2) ? 1.0 + 5.0 * uRing : ((order == 1) ? 1.0 + 0.6 * uRing : 1.0);
  vec3 Sv;
  if (uMode == 2) {
    // redshift map: diverging palette on log g
    float lg = clamp(log2(g) * 0.9, -1.0, 1.0);
    Sv = lg > 0.0 ? mix(vec3(1.0), vec3(0.25, 0.5, 1.6), lg) : mix(vec3(1.0), vec3(1.6, 0.35, 0.12), -lg);
    Sv *= 0.6;
  } else {
    Sv = blackbody(g * T) * uLumNorm;
  }
  float ab = 1.0 - exp(-dtau);
  if (gProbe.w < 0.0 && Tr * ab > 0.02) gProbe = vec4(g, rc, float(order), 1.0);
  I += Tr * ab * Sv * boost * (1.0 - 0.9 * lane * step(0.5, 1.0 - plunge));
  Tr *= exp(-dtau);
}

// sky
// tangent basis
void basis(vec3 d, out vec3 t1, out vec3 t2) {
  vec3 up = abs(d.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  t1 = normalize(cross(up, d));
  t2 = cross(d, t1);
}

// star layer: hashed 3-D cells around the unit sphere, elliptical gaussian filtered by the
// lensing jacobian (pixel footprint on the sky)  -> magnification + aa for free
vec3 starLayer(vec3 d, float cells, float dens, float bright, mat2 Cpix, vec3 t1, vec3 t2, float seed) {
  vec3 p = d * cells;
  vec3 ip = floor(p);
  vec3 f = p - ip;
  vec3 o = step(0.5, f) * 2.0 - 1.0;  // nearest neighbours
  vec3 acc = vec3(0.0);
  float sig2 = pow(uPixAng * 0.42, 2.0);
  for (int k = 0; k < 8; k++) {
    vec3 c = ip + vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1)) * o;
    vec3 h = hash33(c + seed);
    if (h.x > dens) continue;
    vec3 sp = normalize(c + 0.15 + 0.7 * hash33(c + seed + 7.1));
    vec3 dv = sp - d;
    vec2 e = vec2(dot(dv, t1), dot(dv, t2));
    mat2 C = Cpix + mat2(sig2, 0.0, 0.0, sig2);
    float det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
    mat2 Ci = mat2(C[1][1], -C[0][1], -C[1][0], C[0][0]) / det;
    float q = dot(e, Ci * e);
    if (q > 30.0) continue;
    // flux: steep power law (many faint, few bright)
    float fl = bright * pow(max(h.y, 1e-3), -1.1) * 0.03;
    fl = min(fl, 6.0 * bright);
    float temp = mix(2800.0, 14000.0, pow(h.z, 1.8));
    vec3 col = bbChroma(temp);
    acc += col * fl * (uPixAng * uPixAng) * exp(-0.5 * q) / (2.0 * PI * sqrt(det));
  }
  return acc;
}

float fbmSky(vec3 p) {
  float f = 0.0, amp = 0.5;
  for (int i = 0; i < 5; i++) {
    f += amp * (textureLod(uNoise, p, 0.0).r * 2.0 - 1.0);
    p = ROT * p * 2.03 + 0.21;
    amp *= 0.5;
  }
  return f;
}

vec3 sky(vec3 d, mat2 Cpix, vec3 t1, vec3 t2) {
  // galactic band: tilted great circle
  vec3 gN = normalize(vec3(0.12, 0.5, -0.86));     // band passes behind the hole, tilted ~30°
  float lat = dot(d, gN);
  vec3 gC = normalize(vec3(-0.80, 0.52, 0.30));       // galactic centre direction (bulge)
  float cosC = dot(d, gC);
  float band = exp(-pow(lat / 0.16, 2.0));
  float bulge = exp(-pow(lat / 0.22, 2.0)) * pow(max(cosC, 0.0), 6.0);
  float n1 = fbmSky(d * 1.3 + 3.1);
  float n2 = fbmSky(d * 4.1 + 9.7);
  float dust = smoothstep(-0.05, 0.35, n2 + 0.6 * n1) * exp(-pow(lat / 0.07, 2.0));
  float glow = (band * (0.6 + 0.4 * n1) + 1.4 * bulge) * (1.0 - 0.5 * dust);
  vec3 gcol = mix(vec3(0.55, 0.62, 0.85), vec3(1.0, 0.82, 0.62), clamp(0.3 + 0.9 * bulge + 0.3 * n1, 0.0, 1.0));
  vec3 col = gcol * glow * 0.009 * uGalaxy;
  // faint unresolved-star texture in the band
  col += vec3(0.9, 0.85, 0.8) * band * 0.002 * uGalaxy * smoothstep(0.0, 0.6, fbmSky(d * 23.0));

  float bandDens = 1.0 + 2.5 * band * (1.0 - dust);
  vec3 st = vec3(0.0);
  st += starLayer(d, 60.0, 0.045, 1.0, Cpix, t1, t2, 1.3);
  st += starLayer(d, 170.0, 0.035 * bandDens, 0.25, Cpix, t1, t2, 5.7);
  st += starLayer(d, 480.0, 0.03 * bandDens, 0.07, Cpix, t1, t2, 11.9);
  col += st * uStars;
  return col;
}

// main
void main() {
  vec2 frag = uMode == 4 ? uProbe : gl_FragCoord.xy;
  vec2 uv = ((frag + uJitter) / uRes) * 2.0 - 1.0;   // frag = pixel centre, jitter in [-0.5, 0.5]
  float aspect = uRes.x / uRes.y;
  float x = uv.x * aspect * uTanHalf + uShift.x;
  float y = uv.y * uTanHalf + uShift.y;

  // look direction n_cam (zamo frame): forward = −e_r, right = +e_φ, up = −e_θ.
  // the arriving photon travels along m = −n_cam.
  float Nn = inversesqrt(1.0 + x * x + y * y);
  float m_r = Nn, m_th = y * Nn, m_ph = -x * Nn;
  float p_r = uCam.x * m_r;
  float p_th = uCam.y * m_th;
  float L = uCam.z * m_ph;
  float E = uZamo.x + uZamo.y * L;          // E_inf = α E_loc + ω L,  E_loc = 1
  p_r /= E; p_th /= E; L /= E;

  float so = sin(uThO), co = cos(uThO);
  S st;
  st.r = uRo;
  st.pr = p_r;
  st.n = vec3(so, 0.0, co);
  st.J = p_th * vec3(0.0, 1.0, 0.0) - (L / so) * vec3(co, 0.0, -so);

  float rnd = hash12(frag + uSeed * 37.13);
  vec3 I = vec3(0.0);
  float Tr = 1.0;
  int order = 0;
  bool escaped = false;
  bool captured = false;
  bool opaque = false;
  float diskRmax = uRout * 1.15;
  float a2 = uA * uA;

  for (int i = 0; i < 4000; i++) {
    if (i >= uSteps) break;
    S k1 = deriv(st, L);
    float Sig = st.r * st.r + a2 * st.n.z * st.n.z;
    float h = uStepK * min((st.r - uRh) / (abs(k1.r) + 1e-6), 1.0 / (length(k1.n) + 1e-6));

    // disk proximity: limit the affine step so the slab is never skipped
    float s = length(st.n.xy);
    float rc = sqrt(st.r * st.r + a2) * s;
    if (uDiskOn > 0.5 && rc < diskRmax && st.r < diskRmax + 4.0) {
      float H = diskH(rc);
      float az = abs(st.r * st.n.z);
      float dz = az - 3.4 * H;
      // finer steps in the dense mid-plane, coarser in the tenuous upper layers
      float zr = max(az / H - 1.2, 0.0);
      float dsIn = clamp(0.55 * H, 0.012, 0.45) * uDiskStep * (1.0 + 0.5 * zr * zr);
      float hl = max(0.7 * dz, dsIn);
      h = min(h, hl / Sig);
    }
    h = -h;  // backwards in time

    S k2 = deriv(axpy(st, k1, 0.5 * h), L);
    S k3 = deriv(axpy(st, k2, 0.5 * h), L);
    S k4 = deriv(axpy(st, k3, h), L);
    S nx;
    nx.r = st.r + h / 6.0 * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    nx.pr = st.pr + h / 6.0 * (k1.pr + 2.0 * k2.pr + 2.0 * k3.pr + k4.pr);
    nx.n = normalize(st.n + h / 6.0 * (k1.n + 2.0 * k2.n + 2.0 * k3.n + k4.n));
    nx.J = st.J + h / 6.0 * (k1.J + 2.0 * k2.J + 2.0 * k3.J + k4.J);

    // radiative transfer at a jittered point inside the step (stratified -> no banding)
    if (uDiskOn > 0.5) {
      float rs = mix(st.r, nx.r, rnd);
      if (rs < diskRmax + 2.0) {
        vec3 ns = normalize(mix(st.n, nx.n, rnd));
        float zs = rs * ns.z;
        float rcs = sqrt(rs * rs + a2) * length(ns.xy);
        float Hs = diskH(rcs);
        if (abs(zs) < 3.6 * Hs && rcs < diskRmax) {
          float SigS = rs * rs + a2 * ns.z * ns.z;
          float dl = abs(h) * SigS;
          diskSample(rs, ns, mix(st.pr, nx.pr, rnd), L, dl, order, I, Tr);
        }
      }
    }
    // image order: crossings of the equatorial plane inside the disk region
    if (st.n.z * nx.n.z < 0.0 && min(st.r, nx.r) < diskRmax) order++;
    st = nx;

    if (st.r < uRcap) { captured = true; break; }
    if (st.r > uRfar && st.pr < 0.0) { escaped = true; break; }
    if (Tr < 0.004) { opaque = true; break; }
  }

  // escape direction (backwards along the ray), in world frame (camera azimuth applied)
  S kf = deriv(st, L);
  vec3 vel = kf.r * st.n + st.r * kf.n;      // ∝ forward photon velocity (cartesian, large r)
  vec3 dir = normalize(-vel);
  float cp = cos(uPhO), sp = sin(uPhO);
  dir = vec3(cp * dir.x - sp * dir.y, sp * dir.x + cp * dir.y, dir.z);

  // pixel footprint on the sky (uniform control flow here)
  vec3 t1, t2;
  basis(dir, t1, t2);
  vec3 ddx = dFdx(dir), ddy = dFdy(dir);
  vec2 jx = vec2(dot(ddx, t1), dot(ddx, t2));
  vec2 jy = vec2(dot(ddy, t1), dot(ddy, t2));
  float kpx = 0.30;  // pixel filter width (pixels) – accumulation jitter adds the rest
  mat2 Cpix = (outerProduct(jx, jx) + outerProduct(jy, jy)) * kpx * kpx;
  // discontinuities (shadow edge, disk edges): clamp absurd footprints
  Cpix = mat2(min(Cpix[0][0], 0.05), Cpix[0][1], Cpix[1][0], min(Cpix[1][1], 0.05));

  vec3 col;
  if (uMode == 1) {
    col = vec3(escaped ? 1.0 : 0.0);
  } else if (uMode == 3) {
    // ray fate (linear, no tone mapping): escaped dark blue, captured black, absorbed amber, step budget red
    col = escaped ? vec3(0.030, 0.045, 0.085) : captured ? vec3(0.0) : opaque ? vec3(0.95, 0.52, 0.12) : vec3(0.85, 0.06, 0.05);
    if (escaped || opaque) {
      // tint by lensing order: disk-plane crossings n = 1, 2, ≥3 (photon ring)
      vec3 oc = order == 1 ? vec3(0.09, 0.20, 0.40) : order == 2 ? vec3(0.22, 0.48, 0.85) : vec3(0.80, 0.93, 1.0);
      if (order >= 1) col = opaque ? mix(col, oc, 0.35) : oc;
    }
  } else {
    vec3 bg = sky(dir, Cpix, t1, t2);
    col = I + (escaped ? Tr * bg : vec3(0.0));
  }
  col = max(col, vec3(0.0));
  if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);

  if (uMode == 4) {
    // probe: (g or -1, r_emit, order + 10*fate, deflection angle in degrees)
    vec3 look = normalize(-1.0 * vec3(so, 0.0, co) - y * vec3(co, 0.0, -so) + x * vec3(0.0, 1.0, 0.0));
    look = vec3(cp * look.x - sp * look.y, sp * look.x + cp * look.y, look.z);
    float defl = degrees(acos(clamp(dot(look, dir), -1.0, 1.0)));
    float fate = escaped ? 1.0 : (captured ? 2.0 : (opaque ? 3.0 : 4.0));
    outColor = vec4(gProbe.w > 0.0 ? gProbe.x : -1.0, gProbe.y, gProbe.z + 10.0 * fate, escaped ? defl : -1.0);
    return;
  }
  vec3 prev = texelFetch(uPrev, ivec2(frag), 0).rgb;
  outColor = vec4(mix(prev, col, uBlend), 1.0);
}
