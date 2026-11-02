# batch capture of the built page with headless chromium on a gpu node (angle / vulkan), one browser
# for many shots. run from the webgl dir; "cpu" switches to swiftshader.
#   python cluster/batch.py specs.json [page=dist/index.html] [cpu]
#   specs.json = [{"out": "out/x.png", "w": 1600, "h": 900, "hash": "quality=2&...", "samples": 256,
#                  "ui": false, "dpr": 1, "open": "bh,rel", "eval": "", "settle": 0.4, "jpg": 1600}]
#   "jpg" also writes a downscaled jpeg with that long side next to out
import asyncio, json, sys, time, pathlib
from playwright.async_api import async_playwright

ARGS = ['--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-vsync', '--disable-frame-rate-limit',
        '--use-angle=vulkan', '--enable-features=Vulkan', '--use-gl=angle',
        '--disable-background-timer-throttling', '--disable-renderer-backgrounding']


async def shot(browser, s, page_path):
    h = s.get('hash', '')
    if not s.get('ui', False):
        h = (h + '&' if h else '') + 'ui=0&capture=1'
    h = (h + '&' if h else '') + f"samples={s.get('samples', 64)}"
    url = pathlib.Path(page_path).resolve().as_uri() + '#' + h
    pg = await browser.new_page(viewport={'width': s['w'], 'height': s['h']}, device_scale_factor=s.get('dpr', 1))
    pg.set_default_timeout(600000)
    logs = []
    pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}'))
    pg.on('pageerror', lambda e: logs.append(f'pageerror: {e}'))
    t0 = time.time()
    await pg.goto(url)
    try:
        await pg.wait_for_function('window.__kerr && window.__kerr.ready', timeout=60000)
    except Exception:
        for l in logs:
            print('   ', l[:3000], flush=True)
        raise
    for gid in [g for g in s.get('open', '').split(',') if g]:
        await pg.evaluate(f'''() => {{ const s=[...document.querySelectorAll('.group')].find(g=>g.querySelector('h3').textContent.toLowerCase().startsWith('{gid}')); if(s) s.classList.add('open'); }}''')
    if s.get('eval'):
        await pg.evaluate(s['eval'])
    await asyncio.sleep(s.get('settle', 0.4))
    want = s.get('samples', 64)
    while time.time() - t0 < 900:
        n = await pg.evaluate('window.__kerr.samples')
        if n >= want:
            break
        await asyncio.sleep(0.25)
    n = await pg.evaluate('window.__kerr.samples')
    await asyncio.sleep(0.25)
    await pg.screenshot(path=s['out'])
    dt = time.time() - t0
    print(f"{s['out']}: {s['w']}x{s['h']} samples={n} {dt:.1f}s", flush=True)
    for l in logs:
        if 'ERR_FILE_NOT_FOUND' not in l:
            print('   ', l[:3000], flush=True)
    await pg.close()
    if s.get('jpg'):
        from PIL import Image
        im = Image.open(s['out']).convert('RGB')
        L = s['jpg']
        if max(im.size) > L:
            im.thumbnail((L, L), Image.LANCZOS)
        im.save(str(pathlib.Path(s['out']).with_suffix('.jpg')), quality=90)


async def main():
    specs = json.load(open(sys.argv[1]))
    page = sys.argv[2] if len(sys.argv) > 2 else 'dist/index.html'
    args = ARGS
    if len(sys.argv) > 3 and sys.argv[3] == 'cpu':   # swiftshader software gl on a cpu node
        args = [a for a in ARGS if 'angle' not in a and 'Vulkan' not in a] + ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    async with async_playwright() as p:
        b = await p.chromium.launch(args=args, headless=True)
        pg = await b.new_page()
        await pg.goto('about:blank')
        print('renderer:', await pg.evaluate('''() => { const gl=document.createElement('canvas').getContext('webgl2');
             const d=gl.getExtension('WEBGL_debug_renderer_info'); return gl.getParameter(d.UNMASKED_RENDERER_WEBGL); }'''), flush=True)
        await pg.close()
        for s in specs:
            try:
                await shot(b, s, page)
            except Exception as e:
                print('FAILED', s.get('out'), repr(e)[:300], flush=True)
        await b.close()


asyncio.run(main())
