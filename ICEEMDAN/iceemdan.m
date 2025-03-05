function [modes, its] = iceemdan(x, Nstd, NR, MaxIter)
% ICEEMDAN: Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
%
% INPUTs:
% x: signal to decompose
% Nstd: noise standard deviation
% NR: number of realizations
% MaxIter: maximum number of sifting iterations allowed.
%
% OUTPUTs:
% modes: obtained modes in a matrix with rows as the modes
% its: sifting iterations needed for each mode per realization

x = x(:)';
desvio_x = std(x);
x = x / desvio_x;

modes = zeros(size(x));
aux = zeros(size(x));
acum = zeros(size(x));
iter = zeros(NR, round(log2(length(x)) + 5));

% Generate and decompose white noise
white_noise = cell(1, NR);
modes_white_noise = cell(1, NR);
for i = 1:NR
    white_noise{i} = randn(size(x));
    modes_white_noise{i} = emd(white_noise{i});  % Decompose noise with EMD
end

% Compute the first mode using the first IMF of noise
for i = 1:NR
    noise_imf = modes_white_noise{i}(1, :);  % Select the first IMF
    temp = x + Nstd * noise_imf;
    [temp, ~, it] = emd(temp, 'MAXMODES', 1, 'MAXITERATIONS', MaxIter);
    aux = aux + temp(1, :) / NR;
    iter(i, 1) = it;
end

modes = aux; % Save first mode
k = 1;
aux = zeros(size(x));
acum = sum(modes, 1);

while nnz(diff(sign(diff(x - acum)))) > 2  % Compute subsequent modes
    for i = 1:NR
        if size(modes_white_noise{i}, 1) >= k + 1
            noise = modes_white_noise{i}(k, :);  % Select k-th IMF
            noise = noise / std(noise) * Nstd;  % Normalize
            try
                [temp, ~, it] = emd(x - acum + std(x - acum) * noise, 'MAXMODES', 1, 'MAXITERATIONS', MaxIter);
                temp = temp(1, :);
            catch
                it = 0;
                temp = x - acum;
            end
        else
            [temp, ~, it] = emd(x - acum, 'MAXMODES', 1, 'MAXITERATIONS', MaxIter);
            temp = temp(1, :);
        end
        aux = aux + temp / NR;
        iter(i, k + 1) = it;
    end
    modes = [modes; aux];
    aux = zeros(size(x));
    acum = sum(modes, 1);
    k = k + 1;
end

modes = [modes; (x - acum)];
[a, ~] = size(modes);
iter = iter(:, 1:a);
modes = modes * desvio_x;
its = iter;
end