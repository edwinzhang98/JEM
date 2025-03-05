% 获取数据的行数和列数
[num_rows, num_cols] = size(EUA);

% 对每一列进行偏自相关检验并绘制图形
for col = 1:num_cols
    % 取出当前列的数据
    col_data = EUA(:, col);

    % 计算偏自相关系数
    pacf = parcorr(col_data,12);

    % 绘制偏自相关图
    subplot(num_cols, 1, col);  % 用subplot来绘制多个图形，每个图形占据一行
    if col==1
        bar(pacf(2:end),'green')
        grid on;
        ylim([-1, 1]);
        continue
    end
    bar(pacf(2:end),'black');
    grid on;
    ylim([-1, 1]);  % 设置y轴上下限为-1和1

        % 添加0.05和-0.05的虚线
    %hold on;
    %line([0, length(pacf)+1], [0.05, 0.05], 'Color', 'black', 'LineStyle', '--');
    %line([0, length(pacf)+1], [-0.05, -0.05], 'Color', 'black', 'LineStyle', '--');
    %hold off;
    %xlabel('Lag');
    %ylabel('Partial Autocorrelation');
    %title(['Partial Autocorrelation Function (PACF) for Column ', num2str(col)]);
end

% 调整图形布局
%sgtitle('PACF Analysis for Multiple Columns');  % 添加总标题