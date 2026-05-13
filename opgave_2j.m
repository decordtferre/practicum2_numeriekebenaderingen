clear all; clc;
load("BigDatasetCV.mat")
x1 = x;
x2 = y;
b = cat;

n = 3;           % vaste polynoomgraad
n_samples = 50;  % aantal keer samplen per N
N_values = 10 + 20*(1:19);  % N = 30, 50, ..., 390

mean_CV = zeros(length(N_values), 1);
var_CV  = zeros(length(N_values), 1);

N_total = length(x1);

for idx = 1:length(N_values)
    N = N_values(idx);
    CV_estimates = zeros(n_samples, 1);

    parfor s = 1:n_samples
        % Sample N indices zonder terugleggen
        perm = randperm(N_total, N);
        xs1 = x1(perm);
        xs2 = x2(perm);
        bs  = b(perm);

        % LOOCV op deze sample
        CV = zeros(N, 1);
        for i = 1:N
            % Split
            xr = xs1; xr(i) = [];
            yr = xs2; yr(i) = [];
            catr = bs; catr(i) = [];

            % Train
            A = build_A(xr, yr, n);
            [Ntr, M] = size(A);
            func  = @(beta) (1/Ntr) * sum(log(1 + exp(-catr .* (A * beta))));
            Dfunc = @(beta) -(1/Ntr) * (A' * (catr ./ (1 + exp(catr .* (A * beta)))));
            x0 = zeros(M, 1);
            beta = GD(func, Dfunc, x0, 1, 1e-3, 1000);

            % Test
            A_e = build_A(xs1(i), xs2(i), n);
            b_hat = classify(A_e, beta);
            CV(i) = abs(bs(i) - b_hat);
        end

        CV_estimates(s) = mean(CV);
    end

    mean_CV(idx) = mean(CV_estimates);
    var_CV(idx)  = var(CV_estimates);
end

% ===== PLOT =====
figure;

subplot(2,1,1);
plot(N_values, mean_CV, 'b-o', 'LineWidth', 2);
xlabel('N (dataset grootte)');
ylabel('Gemiddelde CV_{LOO} fout');
title('LOOCV schatter: gemiddelde over 50 samples (n=3)');
grid on;

subplot(2,1,2);
plot(N_values, var_CV, 'r-o', 'LineWidth', 2);
xlabel('N (dataset grootte)');
ylabel('Variantie van CV_{LOO} fout');
title('LOOCV schatter: variantie over 50 samples (n=3)');
grid on;

% ===== OPSLAAN =====
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
if ~exist(targetFolder, 'dir'), mkdir(targetFolder); end
fileName = fullfile(targetFolder, [scriptName, '.eps']);
exportgraphics(gcf, fileName, 'ContentType', 'vector');
disp(['Plot opgeslagen als: ', fileName]);