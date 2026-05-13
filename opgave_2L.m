clear all; clc;
load("DatasetCV.mat")
x1 = x; x2 = y; b = cat;
max_n = 12; % gebruik max_n van LOOCV als limiet
N = length(x1);
K = 10;

CV_split = zeros(max_n, 1);
CV_loocv = zeros(max_n, 1);
CV_kfold = zeros(max_n, 1);

% ===== AANPAK 1: Random split =====
% Gemiddelde over meerdere splits voor eerlijkere vergelijking
n_runs = 50;
CV_split_runs = zeros(max_n, n_runs);
parfor r = 1:n_runs
    train_mask = rand(N,1) > 0.5;
    xr = x1(train_mask);  yr = x2(train_mask);  catr = b(train_mask);
    xe = x1(~train_mask); ye = x2(~train_mask);  cate = b(~train_mask);
    for n = 1:max_n
        A = build_A(xr, yr, n);
        [Ntr, M] = size(A);
        func  = @(beta) (1/Ntr) * sum(log(1 + exp(-catr .* (A * beta))));
        Dfunc = @(beta) -(1/Ntr) * (A' * (catr ./ (1 + exp(catr .* (A * beta)))));
        beta  = GD(func, Dfunc, zeros(M,1), 1, 1e-6, 1000);
        A_e   = build_A(xe, ye, n);
        b_hat = classify(A_e, beta);
        CV_split_runs(n,r) = mean(abs(cate - b_hat));
    end
end
CV_split = mean(CV_split_runs, 2);

% ===== AANPAK 2: LOOCV =====
parfor n = 1:max_n
    CV = zeros(N,1);
    for i = 1:N
        xr = x1; xr(i) = [];
        yr = x2; yr(i) = [];
        catr = b; catr(i) = [];
        A = build_A(xr, yr, n);
        [Ntr, M] = size(A);
        func  = @(beta) (1/Ntr) * sum(log(1 + exp(-catr .* (A * beta))));
        Dfunc = @(beta) -(1/Ntr) * (A' * (catr ./ (1 + exp(catr .* (A * beta)))));
        beta  = GD(func, Dfunc, zeros(M,1), 1, 1e-6, 1000);
        A_e   = build_A(x1(i), x2(i), n);
        CV(i) = abs(b(i) - classify(A_e, beta));
    end
    CV_loocv(n) = mean(CV);
end

% ===== AANPAK 3: K-fold =====
idx   = randperm(N);
folds = reshape(idx, K, []);
parfor n = 1:max_n
    CV_local = zeros(K,1);
    for k = 1:K
        test_idx  = folds(k,:);
        train_idx = folds([1:k-1, k+1:end],:); train_idx = train_idx(:);
        xr = x1(train_idx); yr = x2(train_idx); catr = b(train_idx);
        xe = x1(test_idx);  ye = x2(test_idx);  cate = b(test_idx);
        A = build_A(xr, yr, n);
        [Ntr, M] = size(A);
        func  = @(beta) (1/Ntr) * sum(log(1 + exp(-catr .* (A * beta))));
        Dfunc = @(beta) -(1/Ntr) * (A' * (catr ./ (1 + exp(catr .* (A * beta)))));
        beta  = GD(func, Dfunc, zeros(M,1), 1, 1e-6, 1000);
        A_e   = build_A(xe, ye, n);
        CV_local(k) = mean(abs(cate - classify(A_e, beta)));
    end
    CV_kfold(n) = mean(CV_local);
end

% ===== PLOT =====
figure;
plot(1:max_n, CV_split, 'g-o', 'LineWidth', 2); hold on;
plot(1:max_n, CV_kfold, 'b-o', 'LineWidth', 2);
plot(1:max_n, CV_loocv, 'r-o', 'LineWidth', 2);
xlabel('n (polynoomgraad)');
ylabel('Gemiddelde CV-fout');
title('Vergelijking bias: random split vs K-fold vs LOOCV');
legend('Random split (gem. 50 runs)', 'K-fold (K=10)', 'LOOCV (K=N)');
grid on;