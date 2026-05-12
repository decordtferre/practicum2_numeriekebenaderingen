clear all; clc;

load("DatasetCV.mat")

x1 = x;
x2 = y;
b = cat;

max_n = 12;

fouten = zeros(max_n,1);
kruisvalidatiefout = zeros(max_n,1);

N = length(x1);

parfor n = 1:max_n

    CV = zeros(N,1);

    for i = 1:N

        % ===== LOOCV split =====
        xr = x1;
        xr(i) = [];

        yr = x2;
        yr(i) = [];

        catr = b;
        catr(i) = [];

        % ===== TRAIN =====
        A = build_A(xr, yr, n);
        [Ntr, M] = size(A);

        func = @(beta) (1/Ntr) * sum(log(1 + exp(-catr .* (A * beta))));
        Dfunc = @(beta) -(1/Ntr) * (A' * (catr ./ (1 + exp(catr .* (A * beta)))));

        x0 = zeros(M,1);
        beta = GD(func, Dfunc, x0, 1, 1e-6, 1000); % aantal iteraties op 1000 gezet

        % ===== TEST =====
        A_e = build_A(x1(i), x2(i), n);
        b_hat = classify(A_e, beta);

        CV(i) = abs(b(i) - b_hat);

    end

    kruisvalidatiefout(n) = mean(CV);
    fouten(n) = sum(CV ~= 0);

end

% ===== PLOT =====
figure;

subplot(2,1,1);
plot(1:max_n, fouten, 'b-o', 'LineWidth', 2);
xlabel('n'); ylabel('Aantal fouten');
title('LOOCV: testfouten');
grid on;

subplot(2,1,2);
plot(1:max_n, kruisvalidatiefout, 'r-o', 'LineWidth', 2);
xlabel('n'); ylabel('CV fout');
title('LOOCV: kruisvalidatiefout');
grid on;

disp(fouten)

% ===== OPSLAAN VAN PLOT IN /figures =====
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
fileName = fullfile(targetFolder, [scriptName, '.eps']);
exportgraphics(gcf, fileName, 'ContentType', 'vector');
disp(['Plot succesvol opgeslagen als: ', fileName]);