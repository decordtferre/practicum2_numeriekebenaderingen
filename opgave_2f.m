clear all; clc;
load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 20;
aantal_runs = 5;

% Kleuren voor de verschillende runs
kleuren = lines(aantal_runs);
figure; hold on; grid on;

for run = 1:aantal_runs
    % ===== RANDOM DATA SPLIT (train/test) =====
    N_totaal = length(a1);
    indices = randperm(N_totaal);
    split = floor(N_totaal / 2);
    
    idx_tr = indices(1:split);
    idx_te = indices(split+1:end);
    
    xr = a1(idx_tr); yr = a2(idx_tr); catr = b(idx_tr);
    xe = a1(idx_te); ye = a2(idx_te); cate = b(idx_te);
    
    CV = zeros(max_n, 1);
    
    parfor n = 1:max_n
        % Training op Tr
        A_tr = build_A(xr, yr, n);
        [N_tr, M] = size(A_tr);
        
        func = @(beta) (1/N_tr) * sum(log(1 + exp(-catr .* (A_tr * beta))));
        Dfunc = @(beta) -(1/N_tr) * (A_tr' * ((1 ./ (1 + exp(catr .* (A_tr * beta)))) .* catr));
        
        x0 = zeros(M, 1);
        beta = GD(func, Dfunc, x0);
        
        % Testen op Te
        A_te = build_A(xe, ye, n);
        b_hat = classify(A_te, beta);
        
        % CV fout berekening
        CV(n) = (1 / (2 * length(cate))) * sum(abs(cate - b_hat));
    end
    
    plot(1:max_n, CV, '-o', 'Color', kleuren(run,:), 'LineWidth', 1.5, ...
         'DisplayName', ['Run ', num2str(run)]);
end

xlabel('Graad n');
ylabel('Kruisvalidatiefout CV_n');
title('Stabiliteit van de cross-validatiefout over 5 random splits');
legend('Location', 'northeast');
hold off;

% ===== OPSLAAN VAN PLOT IN /figures =====
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
if ~exist(targetFolder, 'dir'), mkdir(targetFolder); end
fileName = fullfile(targetFolder, [scriptName, '.eps']);
exportgraphics(gcf, fileName, 'ContentType', 'vector');