# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'), 
        ('bin', 'bin'),
        ('data', 'data'),
        ('DatasetGuide.md', '.'),
        ('ui', 'ui')
    ],
    hiddenimports=[
        'drawing_canvas', 
        'plot_widget',
        'ui.training_worker',
        'ui.probability_bar_graph',
        'PyQt5.sip',
        'matplotlib.backends.backend_qt5agg'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EasyML',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.png'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EasyML',
)

app = BUNDLE(
    coll,
    name='EasyML.app',
    icon='assets/icon.png',
    bundle_identifier='com.yourdomain.easyml'
)