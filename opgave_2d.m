clear all; clc;

load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 6;

figure;
fouten = zeros(max_n,1);

for n=1:max_n
    % Design matrix bouwen
    A = build_A(a1, a2, n);
    [N,M] = size(A);
    
    % Kostenfunctie en zijn gradiënt
    func = @(beta) (1/N) * sum(log(1 + exp(-b .* (A * beta))));
    Dfunc = @(beta) -(1/N) * (A' * ((1 ./ (1 + exp(b .* (A * beta)))) .* b));

    % Startgok
    x0 = zeros(M,1);

    % GD
    [beta, step] = GD(func, Dfunc, x0);

    % Classificatie
    b_hat = classify(A, beta);

    % Aantal fouten
    fouten(n) = sum(b_hat ~= b);
    
    [xx, yy] = meshgrid(linspace(min(a1)-0.1, max(a1)+0.1, 500), ...
                        linspace(min(a2)-0.1, max(a2)+0.1, 500));

    % Design matrix voor grid
    A_grid = build_A(xx(:), yy(:), n);
    p_grid = 1 ./ (1 + exp(-A_grid * beta));
    p_grid = reshape(p_grid, size(xx));
    
    subplot(2, 3, n)
    hold on;
    grid on;
    xlabel('x1'); ylabel('x2')
    title(sprintf('n = %d | Fouten = %d', n, fouten(n)));
    contour(xx, yy, p_grid, [0.5 0.5], 'k', 'LineWidth', 2);
    scatter(a1(b==1), a2(b==1), 20, 'b', 'filled');
    scatter(a1(b==-1), a2(b==-1), 20, 'r', 'filled');
    hold off
end

disp(fouten)