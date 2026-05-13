clear all; clc;
load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 20;
K = 10;
N = length(a1);
aantal_runs = 5;
kleuren = lines(aantal_runs);

figure; hold on; grid on;

for run = 1:aantal_runs
    % ===== NIEUWE RANDOM SCHUDDING =====
    idx   = randperm(N);
    folds = reshape(idx, K, []);

    CV = zeros(max_n, 1);

    parfor n = 1:max_n
        CV_local = zeros(K, 1);

        for k = 1:K
            test_idx  = folds(k, :);
            train_idx = folds([1:k-1, k+1:end], :);
            train_idx = train_idx(:);

            xr   = x(train_idx); yr   = y(train_idx); catr = cat(train_idx);
            xe   = x(test_idx);  ye   = y(test_idx);  cate = cat(test_idx);

            A = build_A(xr, yr, n);
            [X, M] = size(A);

            func  = @(beta) (1/X) * sum(log(1 + exp(-catr .* (A * beta))));
            Dfunc = @(beta) -(1/X) * (A' * ((1 ./ (1 + exp(catr .* (A * beta)))) .* catr));

            x0   = zeros(M, 1);
            beta = GD(func, Dfunc, x0, 1, 1e-6, 10000);

            A_e  = build_A(xe, ye, n);
            b_hat = classify(A_e, beta);
            CV_local(k) = sum(abs(cate - b_hat));
        end

        CV(n) = sum(CV_local) / N;
    end

    plot(1:max_n, CV, '-o', 'Color', kleuren(run,:), 'LineWidth', 1.5, ...
        'DisplayName', ['Run ', num2str(run)]);
end

xlabel('Graad n');
ylabel('Kruisvalidatiefout CV_n');
legend('Location', 'northeast');
hold off;

% ===== OPSLAAN =====
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
if ~exist(targetFolder, 'dir'), mkdir(targetFolder); end
fileName = fullfile(targetFolder, [scriptName, '.eps']);
exportgraphics(gcf, fileName, 'ContentType', 'vector');
disp(['Plot opgeslagen als: ', fileName]);