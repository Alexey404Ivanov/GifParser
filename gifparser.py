#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIF Parser & CLI Viewer (no external deps)

Goals covered by this file:
- Pure CUI/CLI interface (no GUI libs required)
- Works without graphics modules; optional ANSI terminal preview
- Print detailed info for all headers/blocks (name, raw value, human-readable)
- Optional image show in terminal (ANSI 24-bit) and/or export to PPM
- Per-pixel rendering path (used by both terminal & PPM outputs)
- Correctly handle tiny (1x1) and very large images (downscale-to-fit)
- Basic animation playback in terminal respecting delays & disposal
- 50%-gray checkerboard compositing to visualize transparency

Usage examples:
    python gifparser.py info file.gif
    python gifparser.py show file.gif --max-width 120 --max-height 60
    python gifparser.py animate file.gif --loop 1
    python gifparser.py export-ppm file.gif --out frame_%03d.ppm

Notes:
- Terminal preview uses ANSI truecolor. If your terminal lacks it, use --ascii.
- PPM export writes P6 binary PPM (portable pixmap) that most viewers open.
- This is an educational, clean implementation — no Pillow required.
"""
from __future__ import annotations
import argparse
import sys
import io
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -----------------------------
# Utility: ANSI & helpers
# -----------------------------
CSI = "\x1b["
RESET = CSI + "0m"

def supports_truecolor() -> bool:
    return os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}

# Square block for 2x vertical pixel packing (better aspect in terminal)
BLOCK_UP = "\u2580"  # upper half block
BLOCK_DOWN = "\u2584"  # lower half block
FULL_BLOCK = "\u2588"

# 50%-gray checkerboard colors
CHECKER_LIGHT = (192, 192, 192)
CHECKER_DARK = (128, 128, 128)


def ansi_fg(r: int, g: int, b: int) -> str:
    return f"{CSI}38;2;{r};{g};{b}m"


def ansi_bg(r: int, g: int, b: int) -> str:
    return f"{CSI}48;2;{r};{g};{b}m"

# -----------------------------
# GIF structures
# -----------------------------
class GIFError(Exception):
    pass


def read_u8(buf: io.BufferedReader) -> int:
    b = buf.read(1)
    if not b:
        raise GIFError("Unexpected EOF while reading u8")
    return b[0]


def read_u16le(buf: io.BufferedReader) -> int:
    b = buf.read(2)
    if len(b) < 2:
        raise GIFError("Unexpected EOF while reading u16")
    return b[0] | (b[1] << 8)


def read_bytes(buf: io.BufferedReader, n: int) -> bytes:
    b = buf.read(n)
    if len(b) < n:
        raise GIFError(f"Unexpected EOF while reading {n} bytes")
    return b

@dataclass
class LogicalScreen:
    width: int
    height: int
    gct_flag: bool
    color_resolution: int  # bits per primary - 1 (from packed field)
    sort_flag: bool
    gct_size_exp: int  # size = 2^(N+1)
    bg_color_index: int
    pixel_aspect_ratio: int  # (PAR+15)/64 if PAR != 0

@dataclass
class ImageDescriptor:
    left: int
    top: int
    width: int
    height: int
    lct_flag: bool
    interlace: bool
    sort_flag: bool
    lct_size_exp: int

@dataclass
class GraphicControl:
    disposal_method: int  # 0-7
    user_input: bool
    transparent_flag: bool
    delay_cs: int  # centiseconds
    transparent_index: Optional[int]

@dataclass
class NetscapeLoop:
    loops: int  # 0 means infinite

@dataclass
class Frame:
    descriptor: ImageDescriptor
    lct: Optional[List[Tuple[int,int,int]]]
    lzw_min_code_size: int
    data_subblocks: List[bytes]
    gce: Optional[GraphicControl]

@dataclass
class GIF:
    version: str
    ls: LogicalScreen
    gct: Optional[List[Tuple[int,int,int]]]
    frames: List[Frame]
    netscape: Optional[NetscapeLoop]
    comments: List[str]
    plaintexts: int

# -----------------------------
# Parsing
# -----------------------------

def parse_color_table(buf: io.BufferedReader, size: int) -> List[Tuple[int,int,int]]:
    data = read_bytes(buf, 3*size)
    return [(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]


def parse_gif(path: str) -> GIF:
    with open(path, 'rb') as f:
        buf = io.BufferedReader(f)
        header = read_bytes(buf, 6)
        if header not in (b"GIF87a", b"GIF89a"):
            raise GIFError("Not a GIF file (missing GIF87a/89a)")
        version = header.decode('ascii')

        # Logical Screen Descriptor
        width = read_u16le(buf)
        height = read_u16le(buf)
        packed = read_u8(buf)
        gct_flag = (packed & 0b1000_0000) != 0
        color_resolution = ((packed & 0b0111_0000) >> 4)
        sort_flag = (packed & 0b0000_1000) != 0
        gct_size_exp = (packed & 0b0000_0111)
        bg_color_index = read_u8(buf)
        pixel_aspect_ratio = read_u8(buf)
        ls = LogicalScreen(width, height, gct_flag, color_resolution, sort_flag,
                           gct_size_exp, bg_color_index, pixel_aspect_ratio)
        gct = None
        if gct_flag:
            gct_size = 2 ** (gct_size_exp + 1)
            gct = parse_color_table(buf, gct_size)

        frames: List[Frame] = []
        netscape: Optional[NetscapeLoop] = None
        comments: List[str] = []
        plaintexts = 0
        gce: Optional[GraphicControl] = None

        while True:
            b = buf.read(1)
            if not b:
                raise GIFError("Unexpected EOF before trailer")
            b0 = b[0]
            if b0 == 0x3B:  # Trailer
                break
            elif b0 == 0x21:  # Extension
                label = read_u8(buf)
                if label == 0xF9:  # Graphic Control Extension
                    block_size = read_u8(buf)
                    if block_size != 4:
                        raise GIFError("Bad GCE block size")
                    packed = read_u8(buf)
                    disposal = (packed >> 2) & 0b111
                    user_input = ((packed >> 1) & 1) == 1
                    transparent_flag = (packed & 1) == 1
                    delay_cs = read_u16le(buf)
                    transparent_index = read_u8(buf)
                    terminator = read_u8(buf)
                    if terminator != 0:
                        raise GIFError("Missing GCE block terminator")
                    gce = GraphicControl(disposal, user_input, transparent_flag,
                                         delay_cs, transparent_index if transparent_flag else None)
                elif label == 0xFF:  # Application Extension
                    block_size = read_u8(buf)
                    app_id = read_bytes(buf, block_size)
                    # Read data sub-blocks
                    app_data = b""
                    while True:
                        n = read_u8(buf)
                        if n == 0:
                            break
                        app_data += read_bytes(buf, n)
                    if app_id.startswith(b"NETSCAPE2.0") or app_id.startswith(b"ANIMEXTS1.0"):
                        # Netscape Looping: sub-block "\x01 <loops: u16>"
                        if len(app_data) >= 3 and app_data[0] == 1:
                            loops = app_data[1] | (app_data[2] << 8)
                            netscape = NetscapeLoop(loops)
                elif label == 0xFE:  # Comment Extension
                    # collect comment sub-blocks
                    s = []
                    while True:
                        n = read_u8(buf)
                        if n == 0:
                            break
                        s.append(read_bytes(buf, n))
                    try:
                        comments.append(b"".join(s).decode('utf-8', 'replace'))
                    except Exception:
                        comments.append(repr(b"".join(s)))
                elif label == 0x01:  # Plain Text Extension
                    # Skip Plain Text (rare in practice) but count them
                    block_size = read_u8(buf)
                    read_bytes(buf, block_size)
                    while True:
                        n = read_u8(buf)
                        if n == 0:
                            break
                        read_bytes(buf, n)
                    plaintexts += 1
                else:
                    # Unknown extension — skip sub-blocks
                    while True:
                        n = read_u8(buf)
                        if n == 0:
                            break
                        read_bytes(buf, n)
            elif b0 == 0x2C:  # Image Descriptor
                left = read_u16le(buf)
                top = read_u16le(buf)
                iw = read_u16le(buf)
                ih = read_u16le(buf)
                packed = read_u8(buf)
                lct_flag = (packed & 0b1000_0000) != 0
                interlace = (packed & 0b0100_0000) != 0
                sort_flag = (packed & 0b0010_0000) != 0
                lct_size_exp = (packed & 0b0000_0111)
                lct = None
                if lct_flag:
                    lct_size = 2 ** (lct_size_exp + 1)
                    lct = parse_color_table(buf, lct_size)
                lzw_min_code_size = read_u8(buf)
                data_subblocks: List[bytes] = []
                while True:
                    n = read_u8(buf)
                    if n == 0:
                        break
                    data_subblocks.append(read_bytes(buf, n))
                descriptor = ImageDescriptor(left, top, iw, ih, lct_flag, interlace, sort_flag, lct_size_exp)
                frames.append(Frame(descriptor, lct, lzw_min_code_size, data_subblocks, gce))
                gce = None  # GCE applies to next image only
            else:
                raise GIFError(f"Unknown block introducer: 0x{b0:02X}")

        return GIF(version.decode('ascii'), ls, gct, frames, netscape, comments, plaintexts)

# -----------------------------
# LZW decompression for GIF
# -----------------------------

def lzw_decompress(min_code_size: int, data: bytes) -> List[int]:
    # Build a bitstream reader over GIF sub-block data
    data_bits = []
    for b in data:
        data_bits.append(b)
    # Efficient bit reading without constructing a giant string
    bit_pos = 0
    total_bits = len(data) * 8

    def read_bits(n: int) -> int:
        nonlocal bit_pos
        if bit_pos + n > total_bits:
            return -1
        val = 0
        shift = 0
        while n > 0:
            byte_index = bit_pos // 8
            bit_index = bit_pos % 8
            bits_left_in_byte = 8 - bit_index
            take = min(n, bits_left_in_byte)
            mask = ((1 << take) - 1) << bit_index
            v = (data[byte_index] & mask) >> bit_index
            val |= (v << shift)
            shift += take
            bit_pos += take
            n -= take
        return val

    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    code_mask = (1 << code_size) - 1

    # Initialize dictionary
    dict_size = end_code + 1
    dictionary = {i: [i] for i in range(clear_code)}
    dictionary[clear_code] = []
    dictionary[end_code] = []

    out: List[int] = []
    prev: Optional[List[int]] = None

    while True:
        code = read_bits(code_size)
        if code == -1:
            break
        if code == clear_code:
            # reset
            dictionary = {i: [i] for i in range(clear_code)}
            dictionary[clear_code] = []
            dictionary[end_code] = []
            code_size = min_code_size + 1
            code_mask = (1 << code_size) - 1
            dict_size = end_code + 1
            prev = None
            continue
        if code == end_code:
            break

        if code in dictionary:
            entry = dictionary[code].copy()
        elif prev is not None:
            # KwKwK case
            entry = prev + [prev[0]]
        else:
            raise GIFError("LZW: first code invalid")

        out.extend(entry)

        if prev is not None:
            dictionary[dict_size] = prev + [entry[0]]
            dict_size += 1
            if dict_size == (1 << code_size) and code_size < 12:
                code_size += 1
                code_mask = (1 << code_size) - 1
        prev = entry

    return out

# -----------------------------
# Rasterization & compositing
# -----------------------------

def deinterlace_indices(indices: List[int], w: int, h: int) -> List[int]:
    # GIF 4-pass interlacing
    out = [0]*(w*h)
    i = 0
    for start, step in ((0,8),(4,8),(2,4),(1,2)):
        y = start
        while y < h:
            out[y*w:(y+1)*w] = indices[i:i+w]
            i += w
            y += step
    return out


def decode_frame_pixels(gif: GIF, frame: Frame, canvas_rgba: List[Tuple[int,int,int,int]]) -> Tuple[List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
    # Returns (new_canvas_rgba, frame_rgba_on_transparent)
    # Prepare color table
    ct = frame.lct if frame.lct is not None else gif.gct
    if not ct:
        raise GIFError("Missing color table")

    # Concatenate sub-blocks to a single byte array
    data = b"".join(frame.data_subblocks)
    indices = lzw_decompress(frame.lzw_min_code_size, data)
    w = frame.descriptor.width
    h = frame.descriptor.height
    if frame.descriptor.interlace:
        indices = deinterlace_indices(indices, w, h)
    else:
        if len(indices) < w*h:
            # Pad if truncated (robustness)
            indices = indices + [0]*(w*h - len(indices))

    # Build RGBA for the frame region
    rgba = [(0,0,0,0)]*(w*h)
    t_index = frame.gce.transparent_index if frame.gce else None
    for i, idx in enumerate(indices[:w*h]):
        r,g,b = ct[idx]
        a = 0 if (t_index is not None and idx == t_index) else 255
        rgba[i] = (r,g,b,a)

    # Composite onto canvas considering disposal
    ls = gif.ls
    cw, ch = ls.width, ls.height
    if len(canvas_rgba) != cw*ch:
        canvas_rgba = [(0,0,0,0)]*(cw*ch)

    # Snapshot before drawing (for disposal 3)
    prev_canvas = canvas_rgba.copy()

    # Draw
    for yy in range(frame.descriptor.height):
        for xx in range(frame.descriptor.width):
            src = rgba[yy*w + xx]
            tx = frame.descriptor.left + xx
            ty = frame.descriptor.top + yy
            if 0 <= tx < cw and 0 <= ty < ch:
                dst = canvas_rgba[ty*cw + tx]
                # Alpha over
                sr,sg,sb,sa = src
                if sa == 0:
                    continue
                if dst[3] == 0:
                    canvas_rgba[ty*cw + tx] = (sr,sg,sb,sa)
                else:
                    dr,dg,db,da = dst
                    out_a = sa + (255 - sa)*da//255
                    out_r = (sr*sa + dr*da*(255 - sa)//255)//max(out_a,1)
                    out_g = (sg*sa + dg*da*(255 - sa)//255)//max(out_a,1)
                    out_b = (sb*sa + db*da*(255 - sa)//255)//max(out_a,1)
                    canvas_rgba[ty*cw + tx] = (out_r,out_g,out_b,out_a)

    # Handle disposal method AFTER delay display
    # Return both the displayed frame (composited) and the frame-alone for exporting
    displayed = canvas_rgba.copy()

    disposal = frame.gce.disposal_method if frame.gce else 0
    if disposal == 2:
        # restore to bg color within frame rect
        bg_index = gif.ls.bg_color_index
        bg = (0,0,0,0)
        if gif.gct and 0 <= bg_index < len(gif.gct):
            r,g,b = gif.gct[bg_index]
            bg = (r,g,b,255)
        for yy in range(frame.descriptor.height):
            for xx in range(frame.descriptor.width):
                tx = frame.descriptor.left + xx
                ty = frame.descriptor.top + yy
                if 0 <= tx < cw and 0 <= ty < ch:
                    canvas_rgba[ty*cw + tx] = bg
    elif disposal == 3:
        # restore to previous
        canvas_rgba = prev_canvas

    return displayed, canvas_rgba

# -----------------------------
# Scaling & checkerboard
# -----------------------------

def checker_at(x: int, y: int) -> Tuple[int,int,int]:
    return CHECKER_LIGHT if ((x>>3) ^ (y>>3)) & 1 else CHECKER_DARK


def composite_on_checker(rgba: List[Tuple[int,int,int,int]], w: int, h: int) -> List[Tuple[int,int,int]]:
    out = [(0,0,0)]*(w*h)
    for y in range(h):
        for x in range(w):
            r,g,b,a = rgba[y*w + x]
            br,bg,bb = checker_at(x,y)
            ar = a/255.0
            out[y*w + x] = (int(r*ar + br*(1-ar)), int(g*ar + bg*(1-ar)), int(b*ar + bb*(1-ar)))
    return out


def fit_size(w: int, h: int, max_w: int, max_h: int) -> Tuple[int,int]:
    if w == 0 or h == 0:
        return 0,0
    scale = min(max_w / w, max_h / h)
    if scale >= 1:
        return w, h
    nw = max(1, int(w*scale))
    nh = max(1, int(h*scale))
    return nw, nh


def nearest_downscale(rgb: List[Tuple[int,int,int]], w: int, h: int, nw: int, nh: int) -> List[Tuple[int,int,int]]:
    out = [(0,0,0)]*(nw*nh)
    for y in range(nh):
        sy = int(y*h/nh)
        for x in range(nw):
            sx = int(x*w/nw)
            out[y*nw + x] = rgb[sy*w + sx]
    return out

# -----------------------------
# Terminal rendering
# -----------------------------

def render_terminal(rgb: List[Tuple[int,int,int]], w: int, h: int, ascii_only: bool=False) -> str:
    if not ascii_only and not supports_truecolor():
        ascii_only = True
    out_lines = []
    if ascii_only:
        # crude grayscale ASCII
        chars = " .:-=+*#%@"
        for y in range(h):
            line = []
            for x in range(w):
                r,g,b = rgb[y*w + x]
                lum = (r*299 + g*587 + b*114)//1000
                idx = int(lum*(len(chars)-1)/255)
                line.append(chars[idx])
            out_lines.append("".join(line))
        return "\n".join(out_lines)
    else:
        # pack two vertical pixels per row using LOWER HALF BLOCK (▄)
        # background = top pixel, foreground = bottom pixel
        rows = []
        for y in range(0, h, 2):
            line = []
            for x in range(w):
                top = rgb[y*w + x]
                bottom = rgb[(y+1)*w + x] if y+1 < h else top
                line.append(ansi_bg(*top) + ansi_fg(*bottom) + BLOCK_DOWN)
            line.append(RESET)
            rows.append("".join(line))
        return "\n".join(rows)

# -----------------------------
# PPM export
# -----------------------------

def write_ppm(path: str, rgb: List[Tuple[int,int,int]], w: int, h: int) -> None:
    with open(path, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
        f.write(bytes([c for px in rgb for c in px]))

# -----------------------------
# High-level actions
# -----------------------------

def action_info(gif: GIF) -> None:
    ls = gif.ls
    print("Header:")
    print(f"  Version: {gif.version} (GIF file signature)")
    print("Logical Screen Descriptor:")
    print(f"  Canvas: {ls.width}x{ls.height} pixels (logical screen)")
    print(f"  Global Color Table: {'present' if ls.gct_flag else 'absent'}")
    if ls.gct_flag:
        size = 2 ** (ls.gct_size_exp + 1)
        print(f"    Size: {size} colors; Sorted: {ls.sort_flag}")
        print(f"    Background Color Index: {ls.bg_color_index}")
    print(f"  Color Resolution: {ls.color_resolution + 1} bits per primary")
    if ls.pixel_aspect_ratio != 0:
        par = (ls.pixel_aspect_ratio + 15) / 64.0
        print(f"  Pixel Aspect Ratio: {par:.3f} (height/width)")
    else:
        print("  Pixel Aspect Ratio: not specified (assume square)")

    print(f"Frames: {len(gif.frames)}")
    if gif.netscape:
        loops = gif.netscape.loops
        print(f"  Animation Looping: {'infinite' if loops==0 else loops} times (Netscape ext)")
    if gif.comments:
        print(f"Comments: {len(gif.comments)}")
        for i, c in enumerate(gif.comments[:3], 1):
            snip = (c[:60] + '…') if len(c) > 60 else c
            print(f"  #{i}: {snip}")
    if gif.plaintexts:
        print(f"Plain Text Extensions: {gif.plaintexts}")

    for i, fr in enumerate(gif.frames):
        d = fr.descriptor
        gce = fr.gce
        print(f"\nFrame {i}: {d.width}x{d.height} at ({d.left},{d.top}){' interlaced' if d.interlace else ''}")
        if fr.lct is not None:
            size = 2 ** (d.lct_size_exp + 1)
            print(f"  Local Color Table: present, size {size}")
        if gce:
            disp = gce.disposal_method
            disp_name = {
                0:"None (keep)",1:"Keep (unused)",2:"Restore to BG",3:"Restore to previous"
            }.get(disp, f"Reserved {disp}")
            print(f"  Delay: {gce.delay_cs/100:.2f}s, Transparent idx: {gce.transparent_index}, Disposal: {disp_name}")


def render_first_frame_rgb(gif: GIF, max_w: int, max_h: int) -> Tuple[List[Tuple[int,int,int]], int, int]:
    canvas = [(0,0,0,0)]*(gif.ls.width*gif.ls.height)
    displayed = None
    for fr in gif.frames:
        displayed, canvas = decode_frame_pixels(gif, fr, canvas)
        # Show only the first fully composited frame
        break
    if displayed is None:
        raise GIFError("No frames to render")
    rgb = composite_on_checker(displayed, gif.ls.width, gif.ls.height)
    nw, nh = fit_size(gif.ls.width, gif.ls.height, max_w, max_h)
    if (nw, nh) != (gif.ls.width, gif.ls.height):
        rgb = nearest_downscale(rgb, gif.ls.width, gif.ls.height, nw, nh)
    return rgb, nw, nh


def action_show(gif: GIF, max_w: int, max_h: int, ascii_only: bool=False) -> None:
    rgb, w, h = render_first_frame_rgb(gif, max_w, max_h)
    print(render_terminal(rgb, w, h, ascii_only=ascii_only))


def action_export_ppm(gif: GIF, out_pattern: str, frame_index: Optional[int], max_w: int, max_h: int) -> None:
    canvas = [(0,0,0,0)]*(gif.ls.width*gif.ls.height)
    for i, fr in enumerate(gif.frames):
        displayed, canvas = decode_frame_pixels(gif, fr, canvas)
        if frame_index is not None and i != frame_index:
            continue
        rgb = composite_on_checker(displayed, gif.ls.width, gif.ls.height)
        nw, nh = fit_size(gif.ls.width, gif.ls.height, max_w, max_h)
        if (nw, nh) != (gif.ls.width, gif.ls.height):
            rgb = nearest_downscale(rgb, gif.ls.width, gif.ls.height, nw, nh)
        path = out_pattern % i if ("%" in out_pattern) else out_pattern
        write_ppm(path, rgb, nw, nh)
        if frame_index is not None:
            break


def action_animate(gif: GIF, max_w: int, max_h: int, loop: int, ascii_only: bool=False) -> None:
    canvas = [(0,0,0,0)]*(gif.ls.width*gif.ls.height)
    # Determine total loops
    total_loops = loop
    if loop < 0:
        total_loops = 0  # treat as infinite
    count = 0
    try:
        while total_loops == 0 or count < total_loops:
            for fr in gif.frames:
                displayed, canvas = decode_frame_pixels(gif, fr, canvas)
                rgb = composite_on_checker(displayed, gif.ls.width, gif.ls.height)
                nw, nh = fit_size(gif.ls.width, gif.ls.height, max_w, max_h)
                if (nw, nh) != (gif.ls.width, gif.ls.height):
                    rgb = nearest_downscale(rgb, gif.ls.width, gif.ls.height, nw, nh)
                frame_text = render_terminal(rgb, nw, nh, ascii_only=ascii_only)
                # clear and draw
                sys.stdout.write(CSI + 'H' + CSI + '2J')
                sys.stdout.write(frame_text + "\n")
                sys.stdout.flush()
                delay = (fr.gce.delay_cs/100.0) if fr.gce else 0.1
                # GIF specifies 0 as undefined; use a minimal delay
                if delay <= 0:
                    delay = 0.05
                time.sleep(delay)
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        print(RESET)

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal GIF89a parser & CLI viewer (no deps)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Print detailed header/frame info")
    p_info.add_argument("gif", help="Path to .gif")

    p_show = sub.add_parser("show", help="Render first frame in terminal (ANSI or ASCII)")
    p_show.add_argument("gif")
    p_show.add_argument("--max-width", type=int, default=120)
    p_show.add_argument("--max-height", type=int, default=60)
    p_show.add_argument("--ascii", action="store_true", help="ASCII-only preview (no colors)")

    p_anim = sub.add_parser("animate", help="Play animation in terminal")
    p_anim.add_argument("gif")
    p_anim.add_argument("--max-width", type=int, default=120)
    p_anim.add_argument("--max-height", type=int, default=60)
    p_anim.add_argument("--loop", type=int, default=1, help="Number of loops (0=auto from file, <0=infinite)")
    p_anim.add_argument("--ascii", action="store_true")

    p_ppm = sub.add_parser("export-ppm", help="Export one/all frames to PPM on a 50% gray checkerboard")
    p_ppm.add_argument("gif")
    p_ppm.add_argument("--out", required=True, help="Output path; use %d for frame index")
    p_ppm.add_argument("--frame", type=int, default=None, help="Single frame index (default: all)")
    p_ppm.add_argument("--max-width", type=int, default=4096)
    p_ppm.add_argument("--max-height", type=int, default=4096)

    return p


def main(argv: List[str]) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    try:
        gif = parse_gif(args.gif)
    except GIFError as e:
        print(f"Error: {e}")
        return 2
    except FileNotFoundError:
        print("Error: file not found")
        return 2

    if args.cmd == 'info':
        action_info(gif)
        return 0
    elif args.cmd == 'show':
        # guard huge images by auto-fit
        action_show(gif, max(1, args.max_width), max(1, args.max_height), ascii_only=args.ascii)
        return 0
    elif args.cmd == 'animate':
        loops = args.loop
        if loops == 0 and gif.netscape:
            loops = gif.netscape.loops  # 0 means infinite
        action_animate(gif, args.max_width, args.max_height, loops, ascii_only=args.ascii)
        return 0
    elif args.cmd == 'export-ppm':
        action_export_ppm(gif, args.out, args.frame, args.max_width, args.max_height)
        print("Export done.")
        return 0
    else:
        ap.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
