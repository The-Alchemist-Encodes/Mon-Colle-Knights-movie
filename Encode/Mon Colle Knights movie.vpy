import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import vsutil
import mvsfunc as mvf
import havsfunc as haf
import nnedi3_rpow2
core = vs.core
core.max_cache_size = 8192

src = lvf.src(r"D:\Releases\Sources\Mon Colle Knights movie\Demux\BCBA_0664.d2v")
src = src[::-1] # Don't ask me why this meme works to reduce blending. It's magic.

# Default to TFF if not specified, use opposite frame to restore.
# Use mode 5 and micmatch on every frame, gives the least bad results
deint = core.vivtc.VFM(src, order=1, field=3, mode=5, micmatch=2)
# Fix both leftover combing and some of the less painful frameblending.
# Makes frameblending overall less noticeable during playback.
# Mask is just a lower mthresh version of lvsfunc.deinterlace.decomb's that uses
# the first deinterlace pass rather than the source. Using source left more blends
combmask = core.comb.CombMask(deint, cthresh=1, mthresh=2)
combmask = core.std.Maximum(combmask, threshold=250).std.Maximum(threshold=250).std.Maximum(threshold=250).std.Maximum(threshold=250)
combmask = core.std.BoxBlur(combmask, hradius=2, vradius=2)
# Interpolate over any remaining combing in order to fix it
qtgmc = haf.QTGMC(deint, Preset="Very Slow", EdiMode="nnedi3", NNSize=5, NNeurons=4, InputType=2, TFF=True, SourceMatch=3, Lossless=2, TR0=1, TR1=2, TR2=3, FPSDivisor=1)
# Merge according to the comb mask
qtgmc_merged = core.std.MaskedMerge(deint, qtgmc, combmask)
merged = vsutil.depth(qtgmc_merged, 16)

# Decimate and un-meme
decim = core.vivtc.VDecimate(merged[::-1], chroma=True)
# Split for fixing the chroma shift, yeet everything up to the menu
y,u,v = vsutil.split(decim[1178:])
res_y = core.resize.Spline36(y, height=540)
res_u = core.resize.Spline36(u, height=270, src_left=0.25)
res_v = core.resize.Spline36(v, height=270, src_left=0.25)
resize = vsutil.join([res_y, res_u, res_v])
# Chop off useless stuff shown before the menu and stretch
crop = core.std.CropRel(resize, left=6, right=2, top=4, bottom=0)
# and resize after to prevent fractional resolutions
nn = nnedi3_rpow2.nnedi3_rpow2(crop)

# Mask for some protection, darken the lines against scrubbing
darken = haf.FastLineDarkenMOD(nn, strength=26, protection=4)
mask = core.std.Prewitt(darken)
mask = core.std.Maximum(mask)
mask = vsutil.iterate(mask, core.std.Minimum, 2)

# Denoise the movie, it needs some help, but not too rough
denoise_a = mvf.BM3D(nn, sigma=[2.6,1.9,1.9], radius1=3, profile1="lc", refine=2, full=False)
denoise_b = core.knlm.KNLMeansCL(nn, d=4, a=2, s=2, h=1.0, channels="Y", rclip=denoise_a, device_type="GPU", device_id=1)
denoise_c = core.knlm.KNLMeansCL(nn, d=2, a=1, s=1, h=1.0, channels="UV", rclip=denoise_a, device_type="GPU", device_id=1)
den_y = vsutil.split(denoise_b)[0]
den_u, den_v = vsutil.split(denoise_c)[1:]
denoise = vsutil.join([den_y,den_u,den_v])
denoise = core.std.MaskedMerge(denoise, darken, mask, planes=[0,1,2]).resize.Bicubic(width=denoise.width/2, height=denoise.height/2, filter_param_a=1/3, filter_param_b=1/3)

# Dehaloing, it's bad. Reuse the line and detail mask from before
mask = core.resize.Bicubic(mask, width=mask.width/2, height=mask.height/2, filter_param_a=1/3, filter_param_b=1/3).std.Binarize()
dehalo = haf.FineDehalo(denoise, rx=2.1, ry=1.6, darkstr=0.6, brightstr=0.8)
dehalo = core.std.MaskedMerge(dehalo, denoise, mask)

# Running this after denoise severly lessens the green hue at the top in some scenes.
# For some reason the bottom edge of this thing is fine, but also the most blended
edged = core.edgefixer.ContinuityFixer(dehalo, [4,3,3], [3,3,3], [4,3,3], [0,0,0], 3)

# Debanding and grain before output
deband = core.f3kdb.Deband(denoise, range=13, y=24, cb=16, cr=16, grainy=12, grainc=0, output_depth=16)
grain = kgf.adaptive_grain(deband, 0.3, luma_scaling=8)
out = vsutil.depth(grain, 10)
out.set_output()
