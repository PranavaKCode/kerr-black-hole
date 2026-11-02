# headless chromium capture of the built page (dist/index.html): applies the url hash, waits for
# --samples progressive samples and saves a screenshot. swiftshader by default, so keep local
# captures small; --gpu tries hardware gl.
#   python capture.py out.png --w 480 --h 270 --hash "preset=kerr&quality=0" --samples 4 [--ui] [--gpu]
import argparse, asyncio, time, pathlib

PAGE = pathlib.Path(__file__).resolve().parent / 'dist' / 'index.html'


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--w', type=int, default=480)
    ap.add_argument('--h', type=int, default=270)
    ap.add_argument('--dpr', type=float, default=1.0)
    ap.add_argument('--hash', default='')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--ui', action='store_true', help='keep the UI visible')
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--gpu', action='store_true', help='try hardware GL (EGL/ANGLE) instead of SwiftShader')
    ap.add_argument('--page', default=str(PAGE))
    ap.add_argument('--open', default='', help='comma list of group ids to open (bh,disk,rel,cam,render)')
    ap.add_argument('--eval', default='', help='JS to run before waiting')
    a = ap.parse_args()
    from playwright.async_api import async_playwright
    args = ['--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-vsync', '--disable-frame-rate-limit']
    if a.gpu:
        args += ['--use-angle=vulkan', '--enable-features=Vulkan', '--use-gl=angle']
    else:
        args += ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    h = a.hash
    if not a.ui:
        h = (h + '&' if h else '') + 'ui=0&capture=1'
    h = (h + '&' if h else '') + f'samples={a.samples}'
    url = pathlib.Path(a.page).resolve().as_uri() + '#' + h
    async with async_playwright() as p:
        b = await p.chromium.launch(args=args, headless=True)
        pg = await b.new_page(viewport={'width': a.w, 'height': a.h}, device_scale_factor=a.dpr)
        logs = []
        pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}'))
        pg.on('pageerror', lambda e: logs.append(f'pageerror: {e}'))
        t0 = time.time()
        await pg.goto(url)
        await pg.wait_for_function('window.__kerr && window.__kerr.ready', timeout=60000)
        info = await pg.evaluate('''() => { const gl = document.createElement('canvas').getContext('webgl2');
            const d = gl && gl.getExtension('WEBGL_debug_renderer_info');
            return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : (gl ? gl.getParameter(gl.RENDERER) : 'none'); }''')
        print('renderer:', info, flush=True)
        if a.open:
            for gid in a.open.split(','):
                await pg.evaluate(f'''() => {{ const s=[...document.querySelectorAll('.group')].find(g=>g.querySelector('h3').textContent.toLowerCase().startsWith('{gid}')); if(s) s.classList.add('open'); }}''')
        if a.eval:
            await pg.evaluate(a.eval)
        # wait for samples (the still-mode accumulator starts ~180 ms after the last input)
        last = -1
        while time.time() - t0 < a.timeout:
            s = await pg.evaluate('window.__kerr.samples')
            if s != last:
                print(f'  samples {s}  t={time.time()-t0:.1f}s', flush=True)
                last = s
            if s >= a.samples:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(0.3)
        await pg.screenshot(path=a.out)
        print(f'saved {a.out}  ({time.time()-t0:.1f}s)')
        for l in logs:
            print('  ', l[:2000])
        await b.close()


asyncio.run(main())
