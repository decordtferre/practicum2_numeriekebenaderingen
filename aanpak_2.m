load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 12;

fouten = zeros(max_n,1);

for n=1:max_n
    parfor i = 1:length(a1)
        xr = x;
        xr(i) = [];
        yr = y;
        yr(i) = [];
        catr = cat;
        catr(i) = [];
    % Model trainen
    A = build_A(xr, yr, n);
    beta = GD(A, catr);

    % Testpunt evalueren
    A_e = build_A(x(i), y(i), n);
    b_hat = classify(A_e, beta);
    CV(i) = abs(b(i) - b_hat);
    end

    kruisvalidatiefout(n) = 1/length(a1) * sum(CV);
    fouten(n) = sum(CV ~= 0);
    
end

% Resultaten plotten
figure;
subplot(2,1,1);
plot(1:max_n, fouten, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('n'); ylabel('Aantal fouten');
title('LOOCV: testfouten in functie van n');
grid on;

subplot(2,1,2);
plot(1:max_n, kruisvalidatiefout, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('n'); ylabel('CV-waarde');
title('LOOCV: kruisvalidatiefout in functie van n');
grid on;

disp(fouten)