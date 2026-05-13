clear all; clc;

load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 20;

% ===== TRAININGSDATA =====
xr = zeros(0, 0);
yr = zeros(0, 0);
catr = zeros(0, 0);

% ===== TESTDATA =====
xe = zeros(0, 0);
ye = zeros(0, 0);
cate = zeros(0, 0);

% Random split in trainingsdata en testdata
parfor i = 1:length(a1)
    if randi([0 1]) == 1
        xr = [xr; a1(i)];
        yr = [yr; a2(i)];
        catr = [catr; b(i)];
    else
        xe = [xe; a1(i)];
        ye = [ye; a2(i)];
        cate = [cate; b(i)];
    end
end

fouten = zeros(max_n, 1);
CV = zeros(max_n, 1);

parfor n=1:max_n

    % ===== TRAIN DATA =====
    A = build_A(xr, yr, n);
    [N, M] = size(A);
    func = @(beta) (1/N) * sum(log(1 + exp(-catr .* (A * beta))));
    Dfunc = @(beta) -(1/N) * (A' * ((1 ./ (1 + exp(catr .* (A * beta)))) .* catr));
    x0 = zeros(M, 1);

    beta = GD(func, Dfunc, x0);

    % ===== TEST DATA =====
    A_e = build_A(xe, ye, n);
    b_hat = classify(A_e, beta);
    fouten(n) = sum(b_hat ~= cate);

    CV(n) = sum(abs(cate-b_hat))/N;
end

% ===== PLOT =====
figure;
plot(1:max_n, CV, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('n');
ylabel('Cross-validatie fout');
grid on;

disp(fouten)

% ===== OPSLAAN VAN PLOT IN /figures =====
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
fileName = fullfile(targetFolder, [scriptName, '.eps']);
exportgraphics(gcf, fileName, 'ContentType', 'vector');
disp(['Plot succesvol opgeslagen als: ', fileName]);