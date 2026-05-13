load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 20;
K = 10;
N = length(a1);


% Vouwen maken
idx = randperm(N);
folds = reshape(idx, K, []);

fouten = zeros(max_n, 1);
CV = zeros(max_n, 1);

% Trainen en testen
parfor n=1:max_n
   CV_local = zeros(K, 1);
    fouten_local = zeros(K, 1);

    for k = 1:K
        % Testdata = fold k
        test_idx  = folds(k, :);
        % Trainingsdata = alle andere folds
        train_idx = folds([1:k-1, k+1:end], :);
        train_idx = train_idx(:);

        xr   = x(train_idx);
        yr   = y(train_idx);
        catr = cat(train_idx);
        xe   = x(test_idx);
        ye   = y(test_idx);
        cate = cat(test_idx);

        % Model trainen
        A = build_A(xr, yr, n);
        [X,M] = size(A);

        % Kostenfunctie en zijn gradiënt
        func = @(beta) (1/X) * sum(log(1 + exp(-catr .* (A * beta))));
        Dfunc = @(beta) -(1/X) * (A' * ((1 ./ (1 + exp(catr .* (A * beta)))) .* catr));

        % Startgok
        x0 = zeros(M,1);

        % GD
        [beta, step] = GD(func, Dfunc, x0, 1, 1e-6, 10000);

        % Testfold evalueren
        A_e   = build_A(xe, ye, n);
        b_hat = classify(A_e, beta);

        fouten_local(k) = sum(b_hat ~= cate);
        CV_local(k)     = sum(abs(cate - b_hat));
    end

    fouten(n) = sum(fouten_local);
    CV(n)     = sum(CV_local)/N;  % Gemiddelde over alle K folds
end

figure;
hold on;
plot(1:max_n, CV, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('n'); ylabel('Kruisvalidatiefout');
grid on;

disp(fouten)

% ===== OPSLAAN VAN PLOT IN /figures =====
scriptName = mfilename;
[currentPath, ~, ~] = fileparts(mfilename('fullpath'));
targetFolder = fullfile(currentPath, 'figures');
fileName = fullfile(targetFolder, [scriptName, '.eps']);
exportgraphics(gcf, fileName, 'ContentType', 'vector');
disp(['Plot succesvol opgeslagen als: ', fileName]);