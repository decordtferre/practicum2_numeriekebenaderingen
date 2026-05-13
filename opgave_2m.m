load("BigDatasetCV.mat")
N = 400;
n = 3;
max_K = 15;
n_samples = 200;

x_data   = x;
y_data   = y;
cat_data = cat;

mean_CV = zeros(max_K, 1);
var_CV  = zeros(max_K, 1);

for K = 2:max_K
    CV_samples = zeros(n_samples, 1);

    parfor s = 1:n_samples
        idx_sample = randperm(length(x_data), N);
        xs   = x_data(idx_sample);
        ys   = y_data(idx_sample);
        cats = cat_data(idx_sample);

        idx    = randperm(N);
        N_trim = floor(N / K) * K;
        idx    = idx(1:N_trim);
        folds  = reshape(idx, K, []);
        CV_k   = zeros(K, 1);

        for k = 1:K
            test_idx  = folds(k, :);
            train_idx = folds([1:k-1, k+1:end], :);
            train_idx = train_idx(:);

            xr   = xs(train_idx);
            yr   = ys(train_idx);
            catr = cats(train_idx);
            xe   = xs(test_idx);
            ye   = ys(test_idx);
            cate = cats(test_idx);

            A = build_A(xr, yr, n);
            [X, M] = size(A);

            func  = @(beta) (1/X) * sum(log(1 + exp(-catr .* (A * beta))));
            Dfunc = @(beta) -(1/X) * (A' * ((1 ./ (1 + exp(catr .* (A * beta)))) .* catr));

            x0 = zeros(M, 1);
            [beta, ~] = GD(func, Dfunc, x0, 1, 1e-3, 1000);

            A_e   = build_A(xe, ye, n);
            b_hat = classify(A_e, beta);
            CV_k(k) = sum(abs(cate - b_hat));
        end

        CV_samples(s) = sum(CV_k);
    end

    mean_CV(K) = mean(CV_samples);
    var_CV(K)  = var(CV_samples);
end

K_values = 2:max_K;


% Figuur 2: Variantie van de CV-fout
figure;
plot(K_values, var_CV(2:end), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('K');
ylabel('Variantie van de kruisvalidatiefout');
grid on;

% Opslaan
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
if ~exist(targetFolder, 'dir'), mkdir(targetFolder); end

exportgraphics(figure(1), fullfile(targetFolder, [scriptName, '_gemiddelde.eps']), 'ContentType', 'vector');
exportgraphics(figure(2), fullfile(targetFolder, [scriptName, '_variantie.eps']), 'ContentType', 'vector');
disp(['Plots opgeslagen in: ', targetFolder]);