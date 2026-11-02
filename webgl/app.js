// real-time webgl2 kerr black hole; build.mjs inlines this file, the shaders and the css into dist/index.html
'use strict';
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const DEG = Math.PI / 180;

  // physics helpers
  const Kerr = {
    horizon: a => 1 + Math.sqrt(Math.max(0, 1 - a * a)),
    isco(a) {
      const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
      const z2 = Math.sqrt(3 * a * a + z1 * z1);
      return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
    },
    photon: a => [2 * (1 + Math.cos(2 / 3 * Math.acos(-a))), 2 * (1 + Math.cos(2 / 3 * Math.acos(a)))],
    iscoEL(a, r) {
      const s = Math.sqrt(r), den = Math.pow(r, 0.75) * Math.sqrt(r * s - 3 * s + 2 * a);
      return [(r * s - 2 * s + a) / den, (r * r - 2 * a * s + a * a) / den];
    },
    zamo(a, r, th) {
      const s = Math.sin(th), c = Math.cos(th);
      const Sig = r * r + a * a * c * c, Del = r * r - 2 * r + a * a;
      const A = (r * r + a * a) ** 2 - a * a * Del * s * s;
      return { Sig, Del, A, alpha: Math.sqrt(Sig * Del / A), omega: 2 * a * r / A, gpp: A * s * s / Sig };
    },
    // exact screen (tan) coordinates of the analytic critical curve (bardeen 1973) seen by the
    // zamo camera at (r_o, th_o) — finite distance, same tetrad as the shader.
    criticalCurve(a, ro, tho, n = 720) {
      const z = Kerr.zamo(a, ro, tho);
      const c = Math.cos(tho), s = Math.sin(tho);
      const pts = [];
      const push = (xi, eta, sg) => {
        const Th = eta + a * a * c * c - xi * xi * c * c / (s * s);
        if (Th < 0) return null;
        const R = (ro * ro + a * a - a * xi) ** 2 - z.Del * (eta + (xi - a) ** 2);
        if (R < 0) return null;
        const pth = sg * Math.sqrt(Th), pr = Math.sqrt(R) / z.Del;
        const El = (1 - z.omega * xi) / z.alpha;
        const mr = Math.sqrt(z.Del / z.Sig) * pr / El, mth = pth / (Math.sqrt(z.Sig) * El), mph = xi / (Math.sqrt(z.gpp) * El);
        return [-mph / mr, mth / mr];
      };
      const up = [], dn = [];
      if (a < 1e-3) {
        const b = Math.sqrt(27);
        for (let i = 0; i <= n; i++) {
          const t = i / n * 2 * Math.PI;
          const xi = b * Math.cos(t) * s;
          const p = push(xi, b * b - xi * xi, Math.sin(t) >= 0 ? 1 : -1);
          if (p) pts.push(p);
        }
        return [pts];
      }
      const [rp, rm] = Kerr.photon(a);
      for (let i = 0; i <= n; i++) {
        // cluster samples near the ends where the curve turns quickly
        const u = 0.5 - 0.5 * Math.cos(Math.PI * i / n);
        const r = rp + (rm - rp) * u;
        const xi = (r * r * (3 - r) - a * a * (r + 1)) / (a * (r - 1));
        const eta = r ** 3 * (4 * a * a - r * (r - 3) ** 2) / (a * a * (r - 1) ** 2);
        const p1 = push(xi, eta, 1), p2 = push(xi, eta, -1);
        if (p1) up.push(p1);
        if (p2) dn.push(p2);
      }
      return [up.concat(dn.reverse(), up.length ? [up[0]] : [])];
    },
  };

  // blackbody lut
  // cie 1931 2° observer, multi-lobe gaussian fit (wyman, sloan & shirley 2013) -> xyz -> linear srgb
  function cie(l) {
    const g = (x, mu, s1, s2) => { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
    return [1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2),
      0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1),
      1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)];
  }
  const CIE = []; for (let l = 360; l <= 830; l += 2) CIE.push([l, cie(l)]);
  function bbXYZ(T) {
    let X = 0, Y = 0, Z = 0;
    for (const [l, c] of CIE) {
      const lm = l * 1e-9;
      const B = 1 / (lm ** 5 * (Math.exp(1.438776877e-2 / (lm * T)) - 1));
      X += B * c[0]; Y += B * c[1]; Z += B * c[2];
    }
    return [X, Y, Z];
  }
  const T_MIN = 800, T_MAX = 80000, LUT_N = 512;
  const Y_REF = bbXYZ(6500)[1];
  function bbRGB(T) { // chroma (Y = 1) and log2(Y / Y_6500)
    const [X, Y, Z] = bbXYZ(T);
    let r = (3.2406 * X - 1.5372 * Y - 0.4986 * Z) / Y, g = (-0.9689 * X + 1.8758 * Y + 0.0415 * Z) / Y,
      b = (0.0557 * X - 0.2040 * Y + 1.0570 * Z) / Y;
    const m = Math.min(r, g, b);
    if (m < 0) { const t = -m / (1 - m); r = r + (1 - r) * t; g = g + (1 - g) * t; b = b + (1 - b) * t; }
    return [r, g, b, Math.log2(Y / Y_REF)];
  }
  const LUT = new Float32Array(LUT_N * 4);
  for (let i = 0; i < LUT_N; i++) {
    const T = Math.exp(Math.log(T_MIN) + (Math.log(T_MAX) - Math.log(T_MIN)) * i / (LUT_N - 1));
    LUT.set(bbRGB(T), i * 4);
  }
  function lutAt(T) {
    const u = clamp((Math.log(T) - Math.log(T_MIN)) / (Math.log(T_MAX) - Math.log(T_MIN)), 0, 1) * (LUT_N - 1);
    const i = Math.min(LUT_N - 2, Math.floor(u)), f = u - i;
    const o = [0, 0, 0, 0];
    for (let k = 0; k < 4; k++) o[k] = LUT[i * 4 + k] * (1 - f) + LUT[i * 4 + 4 + k] * f;
    return o;
  }
  function cssColorT(T, scale = 1) {
    const [r, g, b, l2] = lutAt(T);
    const m = Math.max(r, g, b);
    const k = scale / m;
    const enc = c => Math.round(255 * Math.min(1, Math.max(0, c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)));
    return `rgb(${enc(r * k)},${enc(g * k)},${enc(b * k)})`;
  }

  // parameters
  const DEFAULTS = {
    a: 0.99, incl: 81, dist: 60, fov: 31, az: 0, sx: 0, sy: 0,
    iscoLock: 1, rin: 1.4545, rout: 24, thick: 0.02, turb: 1.0, shear: 180, tpeak: 5600, tgamma: 0.45, density: 1.2,
    dust: 0.5, plunge: 0.6, time: 0,
    doppler: 1, grav: 1, ring: 0, curve: 0, view: 0, probe: 1,
    exposure: 0.4, bloom: 0.35, look: 0, sat: 1.6, contrast: 1.25, stars: 1, galaxy: 1, quality: 1, samples: 128, res: 1,
  };
  // presets = the default look + a few overrides
  const LOOK = { az: 0, sx: 0, sy: 0, iscoLock: 1, rout: 24, thick: 0.02, turb: 1.0, shear: 180, tpeak: 5600, tgamma: 0.45,
    density: 1.2, dust: 0.5, plunge: 0.6, ring: 0, exposure: 0.4, view: 0, doppler: 1, grav: 1 };
  const PRESETS = {
    film: { name: 'Gargantua', sub: 'film · Doppler removed', p: { ...LOOK, a: 0.6, incl: 86.5, dist: 70, fov: 22, rout: 26, thick: 0.016, tpeak: 5600, tgamma: 0.5, dust: 0.4, plunge: 0.2, doppler: 0, grav: 0, exposure: 0.0 } },
    kerr: { name: 'Physical Kerr', sub: 'a = 0.99 · full GR', p: { ...LOOK, a: 0.99, incl: 81, dist: 60, fov: 31 } },
    schw: { name: 'Schwarzschild', sub: 'a = 0 · no spin', p: { ...LOOK, a: 0, incl: 81, dist: 60, fov: 30, rout: 26, tpeak: 5600, exposure: -1.2 } },
    face: { name: 'Face-on', sub: 'i = 6° · from the pole', p: { ...LOOK, a: 0.99, incl: 6, dist: 60, fov: 34, turb: 0.9, shear: 70, dust: 0.5, exposure: 1.9 } },
  };

  const QUALITY = [
    { name: 'Draft', steps: 320, k: 0.16, disk: 1.6, oct: 3 },
    { name: 'Balanced', steps: 650, k: 0.11, disk: 1.0, oct: 4 },
    { name: 'High', steps: 1100, k: 0.075, disk: 0.7, oct: 5 },
    { name: 'Ultra', steps: 2000, k: 0.05, disk: 0.45, oct: 5 },
  ];
  const P = Object.assign({}, DEFAULTS);

  // gl setup
  const canvas = $('#gl');
  const overlay = $('#overlay');
  const octx = overlay.getContext('2d');
  const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
  function fail(msg) {
    document.body.classList.add('nogl');
    $('#nogl-msg').textContent = msg;
    throw new Error(msg);
  }
  if (!gl) fail('This piece needs WebGL2, which your browser or device does not provide.');
  const extCBF = gl.getExtension('EXT_color_buffer_float');
  if (!extCBF) fail('Your GPU/browser lacks floating-point render targets (EXT_color_buffer_float).');
  gl.getExtension('OES_texture_float_linear');
  const extTimer = gl.getExtension('EXT_disjoint_timer_query_webgl2');

  const SRC = window.__KERR_SHADERS;
  // programs are compiled in parallel (KHR_parallel_shader_compile) so the page stays responsive while
  // the big tracer compiles (seconds on some windows / d3d drivers); status is checked only when done
  const pext = gl.getExtension('KHR_parallel_shader_compile');
  function startProgram(fsName) {
    const mk = (type, src) => { const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh); return sh; };
    const vs = mk(gl.VERTEX_SHADER, SRC['fullscreen.vert']), fs = mk(gl.FRAGMENT_SHADER, SRC[fsName]);
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.linkProgram(p);
    return { p, vs, fs, name: fsName };
  }
  function finishProgram(o) {
    const { p, vs, fs, name } = o;
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = (gl.getShaderInfoLog(fs) || '') + (gl.getShaderInfoLog(vs) || '') + (gl.getProgramInfoLog(p) || '');
      console.error(name, log);
      fail('Shader compile/link error (' + name + '): ' + log);
    }
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
    return { p, u };
  }
  const PROG_NAMES = ['trace.frag', 'bloom_down.frag', 'bloom_up.frag', 'composite.frag', 'noise_bake.frag'];
  const pendingProgs = PROG_NAMES.map(startProgram);
  let progTrace, progDown, progUp, progComp, progBake;
  function programsReady() {
    return !pext || pendingProgs.every(o => gl.getProgramParameter(o.p, pext.COMPLETION_STATUS_KHR));
  }
  const vao = gl.createVertexArray();

  function tex2D(w, h, internal, format, type, filter, data = null) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function fbo(t) {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) fail('Framebuffer incomplete: 0x' + st.toString(16));
    return f;
  }
  function rt(w, h) { const t = tex2D(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR); return { t, f: fbo(t), w, h }; }
  function freeRT(r) { if (r) { gl.deleteTexture(r.t); gl.deleteFramebuffer(r.f); } }

  // blackbody lut texture
  const bbTex = tex2D(LUT_N, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST, LUT);
  // linear filtering on RGBA32F needs OES_texture_float_linear; fall back to 16F
  let bbTexUse = bbTex;
  if (!gl.getExtension('OES_texture_float_linear')) {
    bbTexUse = tex2D(LUT_N, 1, gl.RGBA16F, gl.RGBA, gl.FLOAT, gl.LINEAR, LUT);
  } else {
    gl.bindTexture(gl.TEXTURE_2D, bbTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  }

  // baked tileable 3-D gradient noise (gpu, once)
  const NOISE_N = 128;
  const noiseTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, noiseTex);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG16F, NOISE_N, NOISE_N, NOISE_N, 0, gl.RG, gl.HALF_FLOAT, null);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.REPEAT], [gl.TEXTURE_WRAP_T, gl.REPEAT], [gl.TEXTURE_WRAP_R, gl.REPEAT]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
  function bakeNoise() {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.useProgram(progBake.p);
    gl.bindVertexArray(vao);
    gl.viewport(0, 0, NOISE_N, NOISE_N);
    gl.uniform1f(progBake.u.uSize, NOISE_N);
    gl.uniform1f(progBake.u.uPeriod, 16);
    for (let z = 0; z < NOISE_N; z++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, noiseTex, 0, z);
      gl.uniform1f(progBake.u.uLayer, z);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.deleteFramebuffer(f);
  }

  // render state
  let accA = null, accB = null, bloomDown = [], bloomUp = [];
  let rw = 0, rh = 0;              // render (trace) resolution
  let cw = 0, ch = 0;              // canvas resolution
  let sampleIdx = 0;               // completed samples in accumulation
  let band = 0, bands = 1;         // tiled progressive rendering
  let dirty = true;                // params changed -> restart accumulation
  let lastInput = -1e9;            // ms
  let dynScale = 0.5;              // adaptive scale while interacting
  let lastMode = '';
  let lastT = 0;
  let pendingQuery = null;
  let gpuMsPerMPix = 0;            // gpu ms per megapixel at current quality (measured)
  let dtMaxSample = 0;             // worst frame interval during the current progressive sample
  let autoRes = 1;                 // automatic still-resolution reduction on very slow gpus
  const MAX_BANDS = 128;

  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

  function resizeCanvas() {
    const w = Math.max(16, Math.round(canvas.clientWidth * dpr()));
    const h = Math.max(16, Math.round(canvas.clientHeight * dpr()));
    if (w !== cw || h !== ch) {
      cw = w; ch = h;
      canvas.width = cw; canvas.height = ch;
      overlay.width = cw; overlay.height = ch;
      dirty = true;
      drawOverlay();
    }
  }
  function ensureTargets(w, h) {
    if (w === rw && h === rh && accA) return;
    freeRT(accA); freeRT(accB);
    for (const b of bloomDown) freeRT(b);
    for (const b of bloomUp) freeRT(b);
    rw = w; rh = h;
    accA = rt(w, h); accB = rt(w, h);
    bloomDown = []; bloomUp = [];
    let bw = Math.max(1, w >> 1), bh = Math.max(1, h >> 1);
    for (let i = 0; i < 7 && bw >= 4 && bh >= 4; i++) {
      bloomDown.push(rt(bw, bh)); bloomUp.push(rt(bw, bh));
      bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
    }
    sampleIdx = 0; band = 0;
  }

  // keep the subject centred in the free area left of the panel (eased when the panel hides)
  // wide screens: panel on the right -> shift left by half its width; phones: bottom sheet -> shift up
  let panelOffsetPx = 0, panelOffsetTarget = 0, panelOffsetYPx = 0, panelOffsetYTarget = 0;
  function updatePanelOffset() {
    const panel = document.getElementById('panel');
    const wide = window.innerWidth > 760;
    const hidden = document.body.classList.contains('hidden-ui');
    const r = panel ? panel.getBoundingClientRect() : null;
    panelOffsetTarget = (wide && !hidden && r) ? (r.width + 16) / 2 : 0;
    panelOffsetYTarget = (!wide && !hidden && r) ? Math.max(0, (window.innerHeight - r.top) / 2) : 0;
  }
  // effective vertical tan(fov/2): P.fov is defined for a 16:9 free area; narrower areas widen the
  // vertical field so the disk keeps its horizontal coverage (capped for portrait phones)
  function camTanHalf() {
    const W = canvas.clientWidth || 1, H = canvas.clientHeight || 1;
    // horizontal coverage of the free area = tanV * W_free / H  (x spans the whole canvas height scale)
    const freeAspect = Math.max(0.2, (W - 2 * panelOffsetPx) / H);
    return Math.tan(P.fov * DEG / 2) * clamp((16 / 9) / freeAspect, 1, 3.0);
  }
  function frameShift() {
    const tanHalf = camTanHalf();
    const W = canvas.clientWidth || 1, H = canvas.clientHeight || 1;
    return panelOffsetPx / (W / 2) * (W / H) * tanHalf;
  }
  function frameShiftY() {
    const H = canvas.clientHeight || 1;
    return -panelOffsetYPx / (H / 2) * camTanHalf();
  }

  // halton jitter
  function halton(i, b) { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; }

  // derived physics
  function derived() {
    const a = clamp(P.a, 0, 0.999);
    const rh_ = Kerr.horizon(a), risco = Kerr.isco(a), [rp, rm] = Kerr.photon(a);
    const rin = P.iscoLock ? risco : Math.max(P.rin, rh_ * 1.02);
    // nt max for the temperature profile used in the shader
    let ntmax = 0;
    for (let i = 0; i <= 400; i++) {
      const x = rin * (1 + 9 * i / 400);
      const f = Math.pow(x, -0.75) * Math.pow(Math.max(1 - 0.92 * Math.sqrt(rin / x), 0), 0.25);
      ntmax = Math.max(ntmax, f);
    }
    const th = clamp(P.incl, 0.5, 179.5) * DEG;
    const ro = P.dist;
    const z = Kerr.zamo(a, ro, th);
    const [Ei, Li] = Kerr.iscoEL(a, risco);
    return { a, rh: rh_, risco, rp, rm, rin, ntmax, th, ro, z, Ei, Li, omegaH: a / (2 * rh_) };
  }

  // trace pass
  function setTraceUniforms(D, mode) {
    const u = progTrace.u;
    const q = QUALITY[mode.quality];
    gl.uniform2f(u.uRes, rw, rh);
    const tanHalf = camTanHalf();
    gl.uniform2f(u.uShift, P.sx + frameShift(), P.sy + frameShiftY());
    gl.uniform1f(u.uTanHalf, tanHalf);
    gl.uniform1f(u.uSeed, mode.seed);
    gl.uniform1i(u.uMode, mode.probe ? 4 : (P.view | 0));
    if (mode.probe) gl.uniform2f(u.uProbe, mode.probe[0], mode.probe[1]);
    gl.uniform1f(u.uA, D.a);
    gl.uniform1f(u.uRh, D.rh);
    gl.uniform1f(u.uRcap, D.rh + Math.min(0.01 * D.rh, 0.3 * (D.rp - D.rh)));
    gl.uniform2f(u.uIsco, D.Ei, D.Li);
    gl.uniform1f(u.uRisco, D.risco);
    gl.uniform1f(u.uRo, D.ro);
    gl.uniform1f(u.uThO, D.th);
    gl.uniform1f(u.uPhO, P.az * DEG);
    gl.uniform3f(u.uCam, Math.sqrt(D.z.Sig / D.z.Del), Math.sqrt(D.z.Sig), Math.sqrt(D.z.gpp));
    gl.uniform2f(u.uZamo, D.z.alpha, D.z.omega);
    gl.uniform1f(u.uRfar, Math.max(2 * D.ro, 100));
    gl.uniform1f(u.uRin, D.rin);
    gl.uniform1f(u.uRout, Math.max(P.rout, D.rin * 1.5));
    gl.uniform1f(u.uThick, P.thick);
    gl.uniform1f(u.uTurb, P.turb);
    gl.uniform1f(u.uShear, P.shear);
    gl.uniform1f(u.uTime, P.time);
    gl.uniform1f(u.uAnimBlend, P.time > 0 ? 180.0 : 0.0);
    gl.uniform1f(u.uDensity, P.density);
    gl.uniform1f(u.uDust, P.dust);
    gl.uniform1f(u.uPlunge, P.plunge);
    gl.uniform1f(u.uTPeak, P.tpeak);
    gl.uniform1f(u.uTGamma, P.tgamma);
    gl.uniform1f(u.uNTmax, D.ntmax);
    gl.uniform1f(u.uDiskOn, (P.view === 1 && !mode.probe) ? 0 : 1);
    gl.uniform1f(u.uDoppler, P.doppler);
    gl.uniform1f(u.uGrav, P.grav);
    gl.uniform1f(u.uRing, P.ring);
    gl.uniform1i(u.uSteps, q.steps);
    gl.uniform1f(u.uStepK, q.k);
    gl.uniform1f(u.uDiskStep, q.disk);
    gl.uniform1i(u.uOctaves, q.oct);
    gl.uniform1f(u.uStars, P.stars);
    gl.uniform1f(u.uGalaxy, P.galaxy);
    gl.uniform1f(u.uLumNorm, 2.8 * Math.pow(2, -lutAt(P.tpeak)[3]));
    gl.uniform1f(u.uPixAng, 2 * tanHalf / Math.max(ch, rh));   // display-pixel angle: star size/flux independent of render scale
    gl.uniform2f(u.uJitter, mode.jx, mode.jy);
    gl.uniform1f(u.uBlend, mode.blend);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, bbTexUse); gl.uniform1i(u.uBB, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, noiseTex); gl.uniform1i(u.uNoise, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, accA.t); gl.uniform1i(u.uPrev, 2);
  }

  function traceBand(D, mode, y0, y1) {
    gl.useProgram(progTrace.p);
    gl.bindVertexArray(vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, accB.f);
    gl.viewport(0, 0, rw, rh);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(0, y0, rw, y1 - y0);
    setTraceUniforms(D, mode);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);
    // copy the band back so accA always holds the full latest accumulation
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, accB.f);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, accA.f);
    gl.blitFramebuffer(0, y0, rw, y1, 0, y0, rw, y1, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  }

  // bloom + composite
  function bloom() {
    if (!bloomDown.length) return null;
    gl.useProgram(progDown.p);
    gl.bindVertexArray(vao);
    let src = accA;
    for (let i = 0; i < bloomDown.length; i++) {
      const d = bloomDown[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.f);
      gl.viewport(0, 0, d.w, d.h);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src.t);
      gl.uniform1i(progDown.u.uSrc, 0);
      gl.uniform2f(progDown.u.uTexel, 1 / src.w, 1 / src.h);
      gl.uniform1f(progDown.u.uKaris, i === 0 ? 1 : 0);
      gl.uniform1f(progDown.u.uClampMax, 400.0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      src = d;
    }
    gl.useProgram(progUp.p);
    let low = bloomDown[bloomDown.length - 1];
    for (let i = bloomDown.length - 2; i >= 0; i--) {
      const d = bloomUp[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.f);
      gl.viewport(0, 0, d.w, d.h);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, low.t); gl.uniform1i(progUp.u.uLow, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, bloomDown[i].t); gl.uniform1i(progUp.u.uCur, 1);
      gl.uniform2f(progUp.u.uTexelLow, 1 / low.w, 1 / low.h);
      gl.uniform2f(progUp.u.uSize, d.w, d.h);
      gl.uniform1f(progUp.u.uRadius, 1.0);
      gl.uniform1f(progUp.u.uMixCur, BLOOM_W[i] || 0.1);
      gl.uniform1f(progUp.u.uLowScale, i === bloomDown.length - 2 ? (BLOOM_W[bloomDown.length - 1] || 0.1) : 1.0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      low = d;
    }
    return low;
  }
  const BLOOM_W = [1.0, 0.85, 0.7, 0.55, 0.42, 0.32, 0.25];
  function composite(frameNo) {
    const b = bloom();
    gl.useProgram(progComp.p);
    gl.bindVertexArray(vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, ch);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, accA.t); gl.uniform1i(progComp.u.uHDR, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, b ? b.t : accA.t); gl.uniform1i(progComp.u.uBloom, 1);
    gl.uniform2f(progComp.u.uSize, cw, ch);
    gl.uniform1f(progComp.u.uExposure, Math.pow(2, P.exposure) * 0.9);
    gl.uniform1f(progComp.u.uBloomAmt, P.view === 1 ? 0 : P.bloom * 0.25);
    gl.uniform1f(progComp.u.uVignette, P.view === 1 ? 0 : 0.28);
    gl.uniform1f(progComp.u.uSat, P.view === 1 ? 1 : 1.3);
    gl.uniform1f(progComp.u.uContrast, P.view === 1 ? 1 : P.contrast);
    gl.uniform1f(progComp.u.uFrame, frameNo);
    let bn = 0; for (let i = 0; i < bloomDown.length; i++) bn += BLOOM_W[i] || 0.1;
    gl.uniform1f(progComp.u.uBloomNorm, bn);
    gl.uniform1f(progComp.u.uSatLin, P.view === 1 ? 1 : P.sat);
    gl.uniform1f(progComp.u.uLook, P.look);
    gl.uniform1f(progComp.u.uRaw, (P.view === 1 || P.view === 3) ? 1 : 0);
    gl.uniform3f(progComp.u.uTint, ...(P.view === 1 ? [1, 1, 1] : [1.0, 0.985, 0.955]));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // main loop
  let frameCount = 0;
  let tweens = null;
  const statusEl = $('#status');
  function frame(t) {
    requestAnimationFrame(frame);
    const dt = Math.min(100, t - lastT); lastT = t;
    frameCount++;
    resizeCanvas();
    stepTween(t);
    pollProbe();
    updatePanelOffset();
    if (Math.abs(panelOffsetPx - panelOffsetTarget) > 0.25 || Math.abs(panelOffsetYPx - panelOffsetYTarget) > 0.25) {
      const k = frameCount < 3 ? 1 : Math.min(1, dt / 120);
      panelOffsetPx += (panelOffsetTarget - panelOffsetPx) * k;
      panelOffsetYPx += (panelOffsetYTarget - panelOffsetYPx) * k;
      if (Math.abs(panelOffsetPx - panelOffsetTarget) <= 0.25) panelOffsetPx = panelOffsetTarget;
      if (Math.abs(panelOffsetYPx - panelOffsetYTarget) <= 0.25) panelOffsetYPx = panelOffsetYTarget;
      dirty = true; lastInput = t; drawOverlay();
    }

    // collect gpu timing of the previous trace
    if (pendingQuery && extTimer) {
      const avail = gl.getQueryParameter(pendingQuery.q, gl.QUERY_RESULT_AVAILABLE);
      const disjoint = gl.getParameter(extTimer.GPU_DISJOINT_EXT);
      if (avail) {
        if (!disjoint) {
          const ms = gl.getQueryParameter(pendingQuery.q, gl.QUERY_RESULT) / 1e6;
          const mpix = pendingQuery.pix / 1e6;
          gpuMsPerMPix = gpuMsPerMPix ? 0.7 * gpuMsPerMPix + 0.3 * (ms / mpix) * (QUALITY[1].steps / QUALITY[pendingQuery.qual].steps) : (ms / mpix) * (QUALITY[1].steps / QUALITY[pendingQuery.qual].steps);
        }
        gl.deleteQuery(pendingQuery.q);
        pendingQuery = null;
      }
    }

    const interacting = (t - lastInput) < 180 || !!tweens;
    const modeName = interacting ? 'interactive' : 'still';
    if (modeName !== lastMode) { dirty = true; lastMode = modeName; }

    // choose resolution
    let scale;
    const userRes = P.res;
    if (modeName === 'interactive') scale = Math.min(dynScale, userRes);
    else scale = userRes * autoRes;
    const w = Math.max(8, Math.round(cw * scale)), h = Math.max(8, Math.round(ch * scale));
    if (w !== rw || h !== rh) { ensureTargets(w, h); dirty = true; }
    if (dirty) { sampleIdx = 0; band = 0; dirty = false; }

    const maxS = modeName === 'still' ? P.samples : Infinity;
    if (sampleIdx >= maxS) { updateStatus(modeName); return; }

    const qual = modeName === 'interactive' ? Math.max(0, P.quality - 1) : P.quality;
    const D = derived();
    const n = sampleIdx;
    const jitter = modeName === 'interactive' ? [0, 0] : [halton(n + 1, 2) - 0.5, halton(n + 1, 3) - 0.5];
    const mode = {
      quality: qual,
      seed: modeName === 'interactive' ? -1 : (n % 997) + 1,
      jx: jitter[0], jy: jitter[1],
      blend: modeName === 'still' ? 1 / (n + 1) : 1,
    };

    // tiled bands for slow gpus in still mode (keep each submit < ~40 ms)
    if (modeName === 'still') dtMaxSample = Math.max(dtMaxSample, dt);
    if (band === 0) {
      if (modeName !== 'still') bands = 1;
      else if (gpuMsPerMPix > 0) {
        const est = gpuMsPerMPix * (w * h / 1e6) * (QUALITY[qual].steps / QUALITY[1].steps);
        bands = clamp(Math.ceil(est / 40), 1, MAX_BANDS);
      } else if (sampleIdx === 0) {
        // before any measurement assume a mid-range gpu (~60 ms per mpix) so the first full-res frame
        // never becomes one huge submit (gpu watchdog / frozen ui on weak devices)
        bands = clamp(Math.ceil(60 * (w * h / 1e6) * (QUALITY[qual].steps / QUALITY[1].steps) / 40), 1, MAX_BANDS);
      } else {
        // no gpu timer: infer from the worst frame interval of the last sample (gpu-bound frames stretch rAF)
        if (dtMaxSample > 70) bands = Math.min(MAX_BANDS, Math.ceil(bands * Math.min(3, dtMaxSample / 45)));
        else if (dtMaxSample < 24 && bands > 1) bands = Math.max(1, Math.floor(bands / 1.3));
      }
      // even the thinnest tiles are too slow -> this is a very weak gpu: lower the still resolution
      if (modeName === 'still' && bands >= MAX_BANDS && sampleIdx >= 2 && (gpuMsPerMPix > 0 ? gpuMsPerMPix * (w * h / 1e6) / MAX_BANDS > 120 : dtMaxSample > 250) && autoRes > 0.35) {
        autoRes = Math.max(0.35, autoRes * 0.7);
        toast('Slow GPU detected — rendering at ' + Math.round(autoRes * 100) + '% resolution (Render → Resolution / Quality)');
      }
      dtMaxSample = 0;
    }
    const y0 = Math.floor(band * h / bands), y1 = Math.floor((band + 1) * h / bands);

    let q = null;
    if (extTimer && !pendingQuery) { q = gl.createQuery(); gl.beginQuery(extTimer.TIME_ELAPSED_EXT, q); }
    traceBand(D, mode, y0, y1);
    if (q) { gl.endQuery(extTimer.TIME_ELAPSED_EXT); pendingQuery = { q, pix: w * (y1 - y0), qual }; }

    band++;
    if (band >= bands) { band = 0; sampleIdx++; }
    composite(frameCount);

    // adapt interactive resolution (target ~ 30 fps)
    if (modeName !== 'still') {
      let est;
      if (gpuMsPerMPix > 0) est = gpuMsPerMPix * (w * h / 1e6) * (QUALITY[qual].steps / QUALITY[1].steps);
      else est = dt;
      const target = 28;
      dynScale = clamp(dynScale * clamp(Math.sqrt(target / Math.max(est, 1)), 0.85, 1.12), 0.18, 1.0);
    }
    updateStatus(modeName);
  }

  function updateStatus(modeName) {
    if (frameCount % 6) return;
    const s = modeName === 'still' ? `${Math.min(sampleIdx, P.samples)} / ${P.samples} samples` : 'interactive';
    statusEl.textContent = `${s} · ${rw}×${rh}${bands > 1 ? ` · ${bands} tiles` : ''}`;
    window.__kerr.samples = sampleIdx;
  }

  // overlay (bardeen curve)
  function drawOverlay() {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!P.curve) return;
    const D = derived();
    const curves = Kerr.criticalCurve(D.a, D.ro, D.th);
    const tanHalf = camTanHalf();
    const W = overlay.width, H = overlay.height;
    const sx = P.sx + frameShift(), sy = P.sy + frameShiftY();
    const toPx = (x, y) => [((x - sx) / (tanHalf * W / H) * 0.5 + 0.5) * W, (0.5 - (y - sy) / tanHalf * 0.5) * H];
    octx.lineWidth = Math.max(1, 1.2 * dpr());
    octx.strokeStyle = 'rgba(120, 220, 255, 0.95)';
    octx.setLineDash([6 * dpr(), 4 * dpr()]);
    for (const c of curves) {
      octx.beginPath();
      c.forEach((p, i) => { const [x, y] = toPx(p[0], p[1]); if (i) octx.lineTo(x, y); else octx.moveTo(x, y); });
      octx.stroke();
    }
    octx.setLineDash([]);
    // label
    const [lx, ly] = toPx(Math.max(...curves[0].map(p => p[0])), 0);
    octx.font = `${11 * dpr()}px Inter, system-ui, sans-serif`;
    octx.fillStyle = 'rgba(120, 220, 255, 0.95)';
    octx.fillText('Bardeen critical curve (analytic)', lx + 10 * dpr(), ly - 6 * dpr());
  }

  // ui
  const fmt = {
    f0: v => v.toFixed(0), f1: v => v.toFixed(1), f2: v => v.toFixed(2), f3: v => v.toFixed(3),
    deg: v => v.toFixed(1) + '°', M: v => v.toFixed(1) + ' M', K: v => Math.round(v).toLocaleString('en-US') + ' K',
    ev: v => (v >= 0 ? '+' : '') + v.toFixed(1) + ' EV', pct: v => Math.round(v * 100) + '%', x: v => v.toFixed(2) + '×',
  };
  const GROUPS = [
    { id: 'bh', title: 'Black hole', open: true, items: [
      { k: 'a', label: 'Spin a', min: 0, max: 0.999, step: 0.001, f: fmt.f3, tip: 'Dimensionless spin a = J/M². 0 = Schwarzschild; ≥ 0.95 flattens the shadow into a “D”.' },
      { type: 'readout', id: 'ro-bh' },
    ] },
    { id: 'disk', title: 'Accretion disk', open: false, items: [
      { k: 'iscoLock', label: 'Inner edge at ISCO', type: 'toggle', tip: 'Lock the inner edge to the innermost stable circular orbit.' },
      { k: 'rin', label: 'Inner radius', min: 1.0, max: 12, step: 0.01, f: fmt.M, tip: 'Inner edge of the disk (gravitational radii).', dis: () => P.iscoLock },
      { k: 'rout', label: 'Outer radius', min: 6, max: 60, step: 0.1, f: fmt.M },
      { k: 'thick', label: 'Thickness H/r', min: 0.005, max: 0.2, step: 0.001, f: fmt.f3, tip: 'Scale height of the vertical Gaussian, relative to radius (slightly flared).' },
      { k: 'tpeak', label: 'Peak temperature', min: 2000, max: 30000, step: 10, f: fmt.K, cls: 'temp', tip: 'Novikov–Thorne-like profile T ∝ r^-3/4 (1−√(r_in/r))^1/4. Real disks peak in X-rays; mapped to visible on purpose.' },
      { k: 'tgamma', label: 'Temperature falloff', min: 0.2, max: 1, step: 0.01, f: fmt.f2, tip: 'Exponent γ of the visible-palette remap T = T_peak·(T_NT/T_max)^γ. 1 = Novikov–Thorne shape (outer disk invisible in the visible); lower = more of the disk glows.' },
      { k: 'density', label: 'Opacity', min: 0.05, max: 12, step: 0.01, f: fmt.f2, tip: 'Absorption coefficient scale: low = translucent glowing gas, high = optically thick.' },
      { k: 'turb', label: 'Turbulence', min: 0, max: 1.6, step: 0.01, f: fmt.f2 },
      { k: 'shear', label: 'Shear (winding)', min: 0, max: 1200, step: 1, f: v => v.toFixed(0) + ' M', tip: 'Age τ of the turbulence in the differentially rotating flow: φ′ = φ − Ω(r)·τ.' },
      { k: 'dust', label: 'Dust lanes', min: 0, max: 1, step: 0.01, f: fmt.pct },
      { k: 'plunge', label: 'Plunging gas', min: 0, max: 1.5, step: 0.01, f: fmt.f2, tip: 'Faint gas spiralling in from the ISCO on geodesics (where frame dragging is strongest).' },
      { k: 'time', label: 'Orbital time t', min: 0, max: 600, step: 0.5, f: v => v.toFixed(0) + ' M', tip: 'Drag to let the gas orbit: Ω = 1/(r^3/2 + a), so the inner disk overtakes the outer disk (differential rotation; frame dragging adds the + a). Turbulence is cross-faded so the flow stays statistically steady.' },
    ] },
    { id: 'rel', title: 'Relativity', open: true, items: [
      { k: 'grav', label: 'Gravitational redshift', type: 'toggle', tip: 'Include 1/u^t (gravitational redshift + time dilation) in g.' },
      { k: 'ring', label: 'Photon-ring emphasis', min: 0, max: 1, step: 0.01, f: fmt.pct, tip: 'Artistic boost of light that orbited the hole (n ≥ 1 images). 0 = physical.' },
      { k: 'probe', label: 'Probe on hover', type: 'toggle', tip: 'Hover the image: traces that one pixel and reports where its light came from (emission radius, redshift g, image order) or how far starlight was bent.' },
      { k: 'curve', label: 'Bardeen critical curve', type: 'toggle', tip: 'Overlay the analytic shadow boundary (Bardeen 1973), mapped through the same camera.' },
      { k: 'view', label: 'View', type: 'seg', opts: ['Beauty', 'Shadow', 'Redshift', 'Rays'], tip: 'Shadow = escaped (white) vs captured (black) rays. Redshift = g map (blue = blueshift). Rays = fate of each ray: escaped (dark; blue tint = crossed the disk plane 1, 2… times), captured (black), absorbed by the disk (amber), step budget exhausted (red).' },
    ] },
    { id: 'cam', title: 'Camera', open: false, items: [
      { k: 'incl', label: 'Inclination', min: 0.5, max: 179.5, step: 0.1, f: fmt.deg, tip: 'Angle from the spin axis. 90° = in the disk plane.' },
      { k: 'az', label: 'Azimuth', min: -180, max: 180, step: 0.1, f: fmt.deg },
      { k: 'dist', label: 'Distance', min: 8, max: 400, step: 0.1, f: fmt.M, log: true, tip: 'Observer radius in M; the camera is a ZAMO (frame-dragged observer).' },
      { k: 'fov', label: 'Field of view', min: 1.5, max: 90, step: 0.1, f: fmt.deg, log: true },
      { k: 'sx', label: 'Lens shift x', min: -0.5, max: 0.5, step: 0.001, f: fmt.f3 },
      { k: 'sy', label: 'Lens shift y', min: -0.5, max: 0.5, step: 0.001, f: fmt.f3 },
      { type: 'button', id: 'reset-cam', label: 'Reset camera' },
    ] },
    { id: 'render', title: 'Render', open: false, items: [
      { k: 'look', label: 'Look', type: 'seg', opts: ['Ember', 'Neutral'], tip: 'Ember: AgX view transform with a warm, saturated grade. Neutral: Khronos PBR Neutral with the Doppler contrast compressed — the grade of the Blender hero still.' },
      { k: 'exposure', label: 'Exposure', min: -5, max: 5, step: 0.05, f: fmt.ev },
      { k: 'bloom', label: 'Bloom', min: 0, max: 1, step: 0.01, f: fmt.pct },
      { k: 'contrast', label: 'Contrast', min: 0.7, max: 1.6, step: 0.01, f: fmt.f2, tip: 'Power applied inside the AgX view transform (Blender’s “Punchy” look ≈ 1.35).' },
      { k: 'sat', label: 'Saturation', min: 0, max: 2.5, step: 0.01, f: fmt.f2, tip: 'Colour saturation of the linear image before the AgX view transform (1 = neutral blackbody colours).' },
      { k: 'stars', label: 'Stars', min: 0, max: 3, step: 0.01, f: fmt.x },
      { k: 'galaxy', label: 'Galactic band', min: 0, max: 3, step: 0.01, f: fmt.x },
      { k: 'quality', label: 'Quality', type: 'seg', opts: QUALITY.map(q => q.name), tip: 'Geodesic step budget and accuracy.' },
      { k: 'samples', label: 'Samples', min: 1, max: 2048, step: 1, f: fmt.f0, log: true, tip: 'Progressive anti-aliasing samples accumulated while the view is still.' },
      { k: 'res', label: 'Resolution', min: 0.25, max: 1, step: 0.01, f: fmt.x },
      { type: 'buttons', items: [{ id: 'shot', label: 'Save PNG' }, { id: 'link', label: 'Copy link' }] },
    ] },
  ];

  const ctl = {};    // key -> {input, out, def}
  function sliderPos(def, v) {
    if (def.log) return Math.log(v / def.min) / Math.log(def.max / def.min);
    return (v - def.min) / (def.max - def.min);
  }
  function sliderVal(def, u) {
    let v = def.log ? def.min * Math.pow(def.max / def.min, u) : def.min + (def.max - def.min) * u;
    if (def.step) v = Math.round(v / def.step) * def.step;
    return clamp(v, def.min, def.max);
  }
  function makeSlider(def, parent, big = false) {
    const row = document.createElement('div');
    row.className = 'row slider' + (big ? ' big' : '') + (def.cls ? ' ' + def.cls : '');
    if (def.tip) row.title = def.tip;
    row.innerHTML = `<div class="lab"><span>${def.label}</span><output></output></div><input type="range" min="0" max="1000" step="1">`;
    parent.appendChild(row);
    const input = row.querySelector('input'), out = row.querySelector('output');
    input.addEventListener('input', () => { set(def.k, sliderVal(def, input.value / 1000)); });
    input.addEventListener('dblclick', () => { set(def.k, DEFAULTS[def.k]); });
    ctl[def.k] = { input, out, def, row };
  }
  function makeToggle(def, parent) {
    const row = document.createElement('label');
    row.className = 'row toggle';
    if (def.tip) row.title = def.tip;
    row.innerHTML = `<span>${def.label}</span><input type="checkbox"><i></i>`;
    parent.appendChild(row);
    const input = row.querySelector('input');
    input.addEventListener('change', () => set(def.k, input.checked ? 1 : 0));
    ctl[def.k] = { input, def, row, toggle: true };
  }
  function makeSeg(def, parent) {
    const row = document.createElement('div');
    row.className = 'row seg';
    if (def.tip) row.title = def.tip;
    row.innerHTML = `<div class="lab"><span>${def.label}</span></div><div class="segs">${def.opts.map((o, i) => `<button data-i="${i}">${o}</button>`).join('')}</div>`;
    parent.appendChild(row);
    row.querySelectorAll('button').forEach(b => b.addEventListener('click', () => set(def.k, +b.dataset.i)));
    ctl[def.k] = { row, def, seg: true };
  }
  function buildUI() {
    // presets
    const pr = $('#presets');
    for (const [id, ps] of Object.entries(PRESETS)) {
      const b = document.createElement('button');
      b.className = 'preset';
      b.dataset.id = id;
      b.innerHTML = `<b>${ps.name}</b><small>${ps.sub}</small>`;
      b.addEventListener('click', () => applyPreset(id));
      pr.appendChild(b);
    }
    // prominent doppler control
    makeSlider({ k: 'doppler', label: 'Doppler & beaming', min: 0, max: 1, step: 0.01, f: v => v.toFixed(2), tip: '0 = film-style Gargantua (Doppler shift and beaming removed, as in Interstellar); 1 = physically correct.' }, $('#doppler'), true);
    const groups = $('#groups');
    for (const g of GROUPS) {
      const sec = document.createElement('section');
      sec.className = 'group' + (g.open ? ' open' : '');
      sec.innerHTML = `<h3><span>${g.title}</span><i class="chev"></i></h3><div class="body"></div>`;
      const body = sec.querySelector('.body');
      sec.querySelector('h3').addEventListener('click', () => sec.classList.toggle('open'));
      for (const it of g.items) {
        if (it.type === 'toggle') makeToggle(it, body);
        else if (it.type === 'seg') makeSeg(it, body);
        else if (it.type === 'readout') { const d = document.createElement('div'); d.className = 'readout'; d.id = it.id; body.appendChild(d); }
        else if (it.type === 'button') { const b = document.createElement('button'); b.className = 'btn'; b.id = it.id; b.textContent = it.label; body.appendChild(b); }
        else if (it.type === 'buttons') { const d = document.createElement('div'); d.className = 'btns'; d.innerHTML = it.items.map(x => `<button class="btn" id="${x.id}">${x.label}</button>`).join(''); body.appendChild(d); }
        else makeSlider(it, body);
      }
      groups.appendChild(sec);
    }
    $('#reset-cam').addEventListener('click', resetCamera);
    $('#shot').addEventListener('click', screenshot);
    $('#link').addEventListener('click', copyLink);
    $('#hide').addEventListener('click', () => document.body.classList.toggle('hidden-ui'));
    $('#about-btn').addEventListener('click', () => document.body.classList.add('about-open'));
    $('#about-close').addEventListener('click', () => document.body.classList.remove('about-open'));
    $('#about').addEventListener('click', e => { if (e.target.id === 'about') document.body.classList.remove('about-open'); });
    syncUI();
  }
  function syncUI() {
    for (const [k, c] of Object.entries(ctl)) {
      const v = P[k];
      if (c.toggle) c.input.checked = !!v;
      else if (c.seg) c.row.querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.i === v));
      else {
        if (document.activeElement !== c.input) c.input.value = Math.round(sliderPos(c.def, v) * 1000);
        c.out.textContent = c.def.f(v);
        c.input.style.setProperty('--p', (sliderPos(c.def, v) * 100).toFixed(2) + '%');
      }
      if (c.def.dis) c.row.classList.toggle('disabled', !!c.def.dis());
    }
    const D = derived();
    if (P.iscoLock) P.rin = D.risco;
    if (ctl.rin) { ctl.rin.out.textContent = fmt.M(D.rin); ctl.rin.input.value = Math.round(sliderPos(ctl.rin.def, D.rin) * 1000); }
    // readouts
    const curve = Kerr.criticalCurve(D.a, 1e4, D.th, 360)[0];
    const xs = curve.map(p => p[0] * 1e4), ys = curve.map(p => p[1] * 1e4);
    const wdt = Math.max(...xs) - Math.min(...xs), hgt = Math.max(...ys) - Math.min(...ys);
    $('#ro-bh').innerHTML = [
      ['Horizon r₊', D.rh.toFixed(3) + ' M'], ['ISCO (prograde)', D.risco.toFixed(3) + ' M'],
      ['Photon orbits', `${D.rp.toFixed(3)} / ${D.rm.toFixed(3)} M`], ['Horizon Ω<sub>H</sub>', D.omegaH.toFixed(4) + ' / M'],
      ['Shadow (w × h)', `${wdt.toFixed(2)} × ${hgt.toFixed(2)} M`],
    ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
    // doppler swatches: inner-disk gas on the approaching / receding limb, same g as the shader
    {
      const r = Math.max(6, D.risco * 2), a = D.a;
      const Om = 1 / (Math.pow(r, 1.5) + a);
      const ut = (Math.pow(r, 1.5) + a) / (Math.pow(r, 0.75) * Math.sqrt(Math.max(1e-6, Math.pow(r, 1.5) - 3 * Math.sqrt(r) + 2 * a)));
      const lam = r * Math.sin(D.th) * 1.15;          // photon leaving the limb towards the camera
      const gG = P.grav ? 1 / ut : 1;
      const gA = gG * Math.pow(1 / (1 - Om * lam), P.doppler), gR = gG * Math.pow(1 / (1 + Om * lam), P.doppler);
      const T = P.tpeak;
      const yA = Math.pow(2, lutAt(T * gA)[3] - lutAt(T)[3]), yR = Math.pow(2, lutAt(T * gR)[3] - lutAt(T)[3]);
      const sw = $('#dswatch');
      const f = v => v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v >= 0.01 ? v.toFixed(2) : v.toPrecision(1);
      if (sw) sw.innerHTML = `<span title="Brightness of the approaching limb relative to the emitted light (visible band)" style="--c:${cssColorT(T * gA)};--b:${Math.min(1, 0.3 + 0.35 * Math.sqrt(yA))}">approaching ×${f(yA)}</span><span title="Brightness of the receding limb relative to the emitted light (visible band)" style="--c:${cssColorT(T * gR)};--b:${Math.min(1, 0.3 + 0.35 * Math.sqrt(yR))}">receding ×${f(yR)}</span>`;
    }
    ctl.tpeak && ctl.tpeak.row.style.setProperty('--tcol', cssColorT(P.tpeak));
    $('#caption').textContent = `a = ${D.a.toFixed(3)}   ·   i = ${P.incl.toFixed(1)}°   ·   r = ${P.dist.toFixed(0)} M   ·   Doppler ${P.doppler.toFixed(2)}`;
    // highlight the preset that matches the current parameters (if any)
    if (!tweens) document.querySelectorAll('.preset').forEach(b => {
      const p = PRESETS[b.dataset.id].p;
      b.classList.toggle('on', Object.keys(p).every(k => k === 'rin' || Math.abs((P[k] ?? 0) - p[k]) < 1e-6));
    });
    updateLegend();
  }
  function updateLegend() {
    const el = $('#legend');
    if (!el) return;
    const v = P.view | 0;
    el.classList.toggle('show', v !== 0);
    if (v === 2) el.innerHTML = '<b>Redshift</b> &nbsp;g = ν<sub>obs</sub> / ν<sub>em</sub> (Doppler × gravity)<div class="bar"></div><div class="ticks"><span>≤ 0.46</span><span>0.7</span><span>1</span><span>1.5</span><span>≥ 2.2</span></div>';
    else if (v === 1) el.innerHTML = '<b>Shadow</b> &nbsp;white: rays that escape to the sky · black: rays captured by the horizon' + (P.curve ? '' : ' · press C for the analytic curve');
    else if (v === 3) el.innerHTML = '<b>Ray fate</b><div class="chips"><span><i style="background:#000"></i>captured</span><span><i style="background:rgb(8,11,22)"></i>escaped, n = 0</span><span><i style="background:rgb(23,51,102)"></i>crossed the disk plane once</span><span><i style="background:rgb(56,122,217)"></i>twice</span><span><i style="background:rgb(204,237,255)"></i>3+ (photon ring)</span><span><i style="background:rgb(242,133,31)"></i>absorbed</span><span><i style="background:rgb(217,15,13)"></i>step budget</span></div>';
  }

  function set(k, v, opts = {}) {
    if (P[k] === v) return;
    P[k] = v;
    dirty = true;
    if (!opts.silent) { lastInput = performance.now(); scheduleHash(); }
    if (k === 'look' && !opts.silent) { // each look comes with its own grade defaults
      const g = v ? { sat: 1.15, contrast: 1.0 } : { sat: 1.6, contrast: 1.25 };
      Object.assign(P, g);
    }
    if (k === 'curve' || ['a', 'incl', 'dist', 'fov', 'sx', 'sy'].includes(k)) drawOverlay();
    syncUI();
  }
  function setMany(obj, silent = false) {
    for (const [k, v] of Object.entries(obj)) P[k] = v;
    dirty = true;
    if (!silent) lastInput = performance.now();
    drawOverlay();
    syncUI();
    scheduleHash();
  }
  function resetCamera() {
    tweenTo({ incl: DEFAULTS.incl, az: 0, dist: DEFAULTS.dist, fov: DEFAULTS.fov, sx: 0, sy: 0 });
  }
  function applyPreset(id) {
    const ps = PRESETS[id];
    tweenTo(ps.p);
  }
  // smooth parameter tweens between presets
  function tweenTo(target, ms = 900) {
    const from = {}, to = {};
    for (const [k, v] of Object.entries(target)) {
      if (typeof v === 'number' && !['view', 'quality', 'iscoLock', 'grav', 'curve', 'look', 'probe'].includes(k)) { from[k] = P[k]; to[k] = v; }
      else P[k] = v;
    }
    if (to.az !== undefined) { // shortest way round
      let d = ((to.az - from.az) % 360 + 540) % 360 - 180; to.az = from.az + d;
    }
    tweens = { from, to, t0: performance.now(), ms };
  }
  function stepTween(t) {
    if (!tweens) return;
    const tw = tweens;
    const u = clamp((t - tweens.t0) / tweens.ms, 0, 1);
    const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
    for (const k of Object.keys(tweens.to)) {
      const a = tweens.from[k], b = tweens.to[k];
      P[k] = (k === 'dist' || k === 'fov' || k === 'tpeak') ? a * Math.pow(b / a, e) : a + (b - a) * e;
    }
    if (u >= 1) { tweens = null; for (const k of Object.keys(tw.to)) P[k] = tw.to[k]; P.az = ((P.az + 540) % 360) - 180; scheduleHash(); }
    dirty = true;
    drawOverlay();
    syncUI();
  }

  // input: orbit / zoom / shift
  let drag = null;
  const pointers = new Map();
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    drag = { x: e.clientX, y: e.clientY, shift: e.shiftKey || e.button === 2, pinch: pointers.size === 2 ? pinchDist() : 0, fov: P.fov };
    tweens = null;
  });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  function pinchDist() { const p = [...pointers.values()]; return Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]); }
  canvas.addEventListener('pointermove', e => {
    if (!drag || !pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) {
      const d = pinchDist();
      if (drag.pinch) set('fov', clamp(drag.fov * drag.pinch / d, 1.5, 90));
      return;
    }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.shift) {
      const k = 2 * camTanHalf() / canvas.clientHeight;
      setMany({ sx: clamp(P.sx - dx * k, -0.5, 0.5), sy: clamp(P.sy + dy * k, -0.5, 0.5) });
    } else {
      const k = 0.18 * Math.sqrt(P.fov / 22);
      setMany({ az: ((P.az - dx * k + 540) % 360) - 180, incl: clamp(P.incl - dy * k, 0.5, 179.5) });
    }
  });
  // hover probe: what is this pixel?
  const probeTex = tex2D(1, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
  const probeFbo = fbo(probeTex);
  const probeOut = new Float32Array(4);
  const probeEl = document.getElementById('probe');
  const probePbo = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, probePbo);
  gl.bufferData(gl.PIXEL_PACK_BUFFER, 16, gl.STREAM_READ);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  let probeAt = null, probeT = 0, probeSync = null, probeShownAt = null, probeQueued = false;
  // asynchronous readback (pbo + fence) so hovering never stalls the gpu pipeline
  function pollProbe() {
    if (!probeSync) return;
    const st = gl.clientWaitSync(probeSync, 0, 0);
    if (st === gl.TIMEOUT_EXPIRED) return;
    gl.deleteSync(probeSync); probeSync = null;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, probePbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, probeOut);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    if (probeAt) showProbe(probeShownAt);
    if (probeQueued) { probeQueued = false; runProbe(); }
  }
  function runProbe() {
    if (!probeAt || !accA || !P.probe) return;
    if (probeSync) { probeQueued = true; return; }   // one readback in flight at a time
    const [cx, cy] = probeAt;
    const px = cx * dpr() * (rw / cw), py = (ch - cy * dpr()) * (rh / ch);
    const D = derived();
    gl.useProgram(progTrace.p);
    gl.bindVertexArray(vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, probeFbo);
    gl.viewport(0, 0, 1, 1);
    setTraceUniforms(D, { quality: Math.max(P.quality, 2), seed: 1, jx: 0, jy: 0, blend: 1, probe: [px, py] });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, probePbo);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    probeSync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    probeShownAt = [cx, cy];
  }
  function showProbe(at) {
    const [cx, cy] = at;
    const [g, r, of, defl] = probeOut;
    const fate = Math.floor((of + 0.5) / 10), order = Math.round(of - 10 * fate);
    let html;
    if (g > 0) {
      const shift = g > 1.02 ? 'blueshifted' : g < 0.98 ? 'redshifted' : 'unshifted';
      html = `<b>Disk light</b> emitted at r = ${r.toFixed(2)} M<br>g = ν<sub>obs</sub>/ν<sub>em</sub> = ${g.toFixed(2)} · ${shift}<br>image order n = ${order}${order >= 2 ? ' (photon ring)' : order === 1 ? ' (lensed around the hole)' : ''}`;
    } else if (fate === 1) html = `<b>Starlight</b><br>bent by ${defl.toFixed(1)}° on its way past the hole`;
    else if (fate === 2) html = '<b>Captured</b><br>this ray falls through the horizon';
    else html = '<b>Still orbiting</b><br>ray circles the photon region beyond the step budget';
    probeEl.innerHTML = html;
    probeEl.style.transform = `translate(${Math.min(cx + 16, innerWidth - 250)}px, ${Math.min(cy + 16, innerHeight - 90)}px)`;
    probeEl.classList.add('show');
  }
  canvas.addEventListener('pointermove', e => {
    if (drag || e.pointerType === 'touch') { probeEl.classList.remove('show'); return; }
    probeAt = [e.clientX, e.clientY];
    const now = performance.now();
    if (now - probeT > 90) { probeT = now; runProbe(); }
  });
  canvas.addEventListener('pointerleave', () => { probeAt = null; probeEl.classList.remove('show'); });
  canvas.addEventListener('pointerdown', () => probeEl.classList.remove('show'));

  const endDrag = e => { pointers.delete(e.pointerId); if (!pointers.size) drag = null; };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    tweens = null;
    const f = Math.exp(clamp(e.deltaY, -200, 200) * 0.0012);
    if (e.shiftKey) set('dist', clamp(P.dist * f, 8, 400));
    else set('fov', clamp(P.fov * f, 1.5, 90));
  }, { passive: false });
  canvas.addEventListener('dblclick', resetCamera);
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.key === 'h' || e.key === 'H') document.body.classList.toggle('hidden-ui');
    else if (e.key === 's' || e.key === 'S') screenshot();
    else if (e.key === 'r' || e.key === 'R') resetCamera();
    else if (e.key === 'c' || e.key === 'C') set('curve', P.curve ? 0 : 1);
    else if (e.key === 'Escape') document.body.classList.remove('about-open');
    else if ('1234'.includes(e.key)) applyPreset(Object.keys(PRESETS)[+e.key - 1]);
  });

  // share / screenshot
  const HASH_KEYS = Object.keys(DEFAULTS);
  let hashTimer = 0;
  function scheduleHash() {
    clearTimeout(hashTimer);
    hashTimer = setTimeout(() => {
      const parts = [];
      for (const k of HASH_KEYS) if (Math.abs(P[k] - DEFAULTS[k]) > 1e-9 && !(k === 'rin' && P.iscoLock)) parts.push(k + '=' + (+P[k].toFixed(4)));
      history.replaceState(null, '', parts.length ? '#' + parts.join('&') : location.pathname + location.search);
    }, 300);
  }
  function readHash() {
    const h = location.hash.slice(1);
    if (!h) return;
    const o = {};
    for (const kv of h.split('&')) {
      const [k, v] = kv.split('=');
      if (k === 'preset' && PRESETS[v]) Object.assign(o, PRESETS[v].p);
      else if (k in DEFAULTS && v !== undefined && isFinite(+v)) o[k] = +v;
      else if (k === 'ui' && v === '0') document.body.classList.add('hidden-ui');
      else if (k === 'capture') document.body.classList.add('capture');
    }
    Object.assign(P, o);
  }
  function screenshot() {
    canvas.toBlob(b => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = `kerr_a${P.a.toFixed(3)}_i${P.incl.toFixed(0)}_dop${P.doppler.toFixed(2)}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }, 'image/png');
  }
  function copyLink() {
    const url = location.href;
    (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject()).then(
      () => toast('Link with current view copied'), () => toast(url));
  }
  let toastT = 0;
  function toast(msg) {
    const t = $('#toast'); t.textContent = msg; t.classList.add('show');
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 2200);
  }

  // go
  window.__kerr = { P, set, setMany, samples: 0, ready: false };   // read by capture.py / cluster/batch.py
  readHash();
  buildUI();
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  function boot() {
    if (!programsReady()) { requestAnimationFrame(boot); return; }
    [progTrace, progDown, progUp, progComp, progBake] = pendingProgs.map(finishProgram);
    bakeNoise();
    document.body.classList.add('booted');
    // open with a quick low-resolution preview, then refine progressively to a still
    lastInput = performance.now() - 60;
    requestAnimationFrame(t => { lastT = t; window.__kerr.ready = true; frame(t); });
  }
  requestAnimationFrame(boot);
})();
