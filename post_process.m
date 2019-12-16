%% Post processing of merged RAW
merged_raw = imread("./merged_raw_crowd.tiff");

f_raw = double(merged_raw) / 65535.0;

[H, W] = size(f_raw);

%% White Balance
wb_mults = [1.8285714387893677, 1.0, 1.5609755516052246, 1.0]; % crowd
% wb_mults = [1.802815318107605, 1.0, 1.7297297716140747, 1.0]; % flowers
% wb_mults = [1.4566144943237305, 1.0, 2.202150583267212, 1.0]; % WH
% wb_mults = [1.2, 1, 1.1, 1];
% df_raw = zeros(size(f_raw));
% df_raw(1:2:end, 1:2:end) = f_raw(1:2:end, 1:2:end) * wb_mults(1);
% df_raw(1:2:end, 2:2:end) = f_raw(1:2:end, 2:2:end) * wb_mults(2);
% df_raw(2:2:end, 1:2:end) = f_raw(2:2:end, 1:2:end) * wb_mults(2);
% df_raw(2:2:end, 2:2:end) = f_raw(2:2:end, 2:2:end) * wb_mults(3);



%% Demosaic

% Split Image into 3 channels
ch_r = f_raw(1:2:end, 1:2:end);
ch_g1 = f_raw(1:2:end, 2:2:end);
ch_g2 = f_raw(2:2:end, 1:2:end);
ch_b = f_raw(2:2:end, 2:2:end);

ch_r = ch_r * wb_mults(3);
ch_g1 = ch_g1 * wb_mults(2);
ch_g2 = ch_g2 * wb_mults(2);
ch_b = ch_b * wb_mults(1);

[X1, Y1] = meshgrid(1:W/2, 1:H/2);
[X2, Y2] = meshgrid(1:W, 1:H);

ch_r = interp2(X1, Y1, ch_r, X2 / 2, Y2 / 2, 'bilinear', 0);
ch_g1 = interp2(X1, Y1, ch_g1, X2 / 2, Y2 / 2, 'bilinear', 0);
ch_g2 = interp2(X1, Y1, ch_g2, X2 / 2, Y2 / 2, 'bilinear', 0);
ch_b = interp2(X1, Y1, ch_b, X2 / 2, Y2 / 2, 'bilinear', 0);

ch_g = (ch_g1 + ch_g2) ./ 2;

img = cat(3, ch_r, ch_g, ch_b);

%% Chroma denoising

% Denoise the color channels in Lab space 
lab_img = rgb2lab(img, 'ColorSpace', 'linear');

% Bilateral filtering for denoising
lab_img(:, :, 2) = imbilatfilt(lab_img(:, :, 2), 0.1);
lab_img(:, :, 3) = imbilatfilt(lab_img(:, :, 3), 0.1);

rgb_img = lab2rgb(lab_img, 'ColorSpace', 'linear');

%% Color Correction
cc_matrix = [1.7734375   -0.765625   -0.0078125; 
            -0.2578125   1.5078125   -0.25; 
               0         -0.7265625   1.7265625];
           
% cc_matrix = eye(3);
           
% cc_matrix = [1.7265625 -0.703125 -0.0234375; -0.28125 1.4609375 -0.1796875; 0.015625 -0.7109375 1.6953125];

 cc_img = color_calibrate(rgb_img, cc_matrix);
 cc_img(cc_img < 0) = 0;

%% Merge Multi exposure
exp_1 = imadjust(cc_img * 100, [], [], 1);
exp_2 = imadjust(cc_img * 200, [], [], 1);
exp_3 = imadjust(cc_img * 300, [], [],  1);
exp_4 = imadjust(cc_img * 400, [], [],  1);
exp_5 = imadjust(cc_img * 500, [], [],  1);
exp_6 = imadjust(cc_img * 600, [], [],  1);
exp_7 = imadjust(cc_img * 700, [], [],  1);
b_img = blendexposure(exp_1, exp_2, exp_3, exp_4, exp_5, exp_6, exp_7);

%%
final_img = localtonemap(single(imlocalbrighten(cc_img * 100, 'AlphaBlend', true)));

lab_img_final = rgb2lab(final_img);
lab_img_final(: ,:, 1) = adapthisteq(lab_img_final(:, :, 1) / 100);
new_final_img = lab2rgb(lab_img_final);





%% Local Tonemap

%% Chroma denoising

% Denoise the color channels in Lab space 
lab_img = rgb2lab(final_img);

% Bilateral filtering for denoising
lab_img(:, :, 2) = imbilatfilt(lab_img(:, :, 2), 1);
lab_img(:, :, 3) = imbilatfilt(lab_img(:, :, 3), 1);

last_img = lab2rgb(lab_img);

%% Global tonemap
net = denoisingNetwork('DnCNN');
final_img(:, :, 1) = denoiseImage(final_img(:, :, 1), net);
final_img(:, :, 2) = denoiseImage(final_img(:, :, 2), net);
final_img(:, :, 3) = denoiseImage(final_img(:, :, 3), net);

%% Gamma Correction



