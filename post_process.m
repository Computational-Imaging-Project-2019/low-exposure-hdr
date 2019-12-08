%% Post processing of merged RAW
merged_raw = imread("./merged_raw.tiff");

f_raw = double(merged_raw) / 65535.0;

[H, W] = size(f_raw);

%% White Balance
% wb_mults = [1.8285714387893677, 1.0, 1.5609755516052246, 1.0];

%% Demosaic

% Split Image into 3 channels
ch_b = f_raw(1:2:end, 1:2:end);
ch_g1 = f_raw(1:2:end, 2:2:end);
ch_g2 = f_raw(2:2:end, 1:2:end);
ch_r = f_raw(2:2:end, 2:2:end);

% ch_b = ch_b * wb_mults(1);
% ch_g1 = ch_g1 * wb_mults(2);
% ch_g2 = ch_g2 * wb_mults(3);
% ch_r = ch_r * wb_mults(4);

[X1, Y1] = meshgrid(1:W/2, 1:H/2);
[X2, Y2] = meshgrid(1:W, 1:H);

ch_r = interp2(X1, Y1, ch_r, X2 / 2, Y2 / 2, 'spline', 0);
ch_g1 = interp2(X1, Y1, ch_g1, X2 / 2, Y2 / 2, 'spline', 0);
ch_g2 = interp2(X1, Y1, ch_g2, X2 / 2, Y2 / 2, 'spline', 0);
ch_b = interp2(X1, Y1, ch_b, X2 / 2, Y2 / 2, 'spline', 0);

ch_g = (ch_g1 + ch_g2) ./ 2;

img = cat(3, ch_r, ch_g, ch_b);

img(img < 0) = 0;

%% Chroma denoising

%% Color Correction

%% Merge Multi exposure
b_img = blendexposure(img, img * 50, img * 300,img * 500, img * 600);

%% Local Tonemap

%% Global tonemap

%% Gamma Correction


