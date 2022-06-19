for i = 3 : 20
    avg = real(example_vasarhelyi(1, 1, i));
    if ~isreal(avg)
        avg = 0;
    end
%     end
    disp(int2str(i));
    disp(avg);
    writematrix(avg,(strcat(int2str(i),'_gamer.csv')));
end


% final_res = zeros(1, 20, 20);
% %for i = 1 : 20
% %    final_res(i) = zeros(1, 20);
% %end
% avg = zeros(1,20);
% for i = 0.70 : 0.05 : 1
% % for i = 0.35 : 0.05 : 1
%     ix = floor(i*20) + 1;
% 
%     for j = 0 : 0.05 : 1
%         avg = zeros(1,20);
%  
%         for k = 1 : 20
% %         %avg = [avg, example_vasarhelyi(i)];
%             avg(k) = example_vasarhelyi(i, j);
%         end
% % 
%         jx = floor(j*20) + 1;
%         disp(strcat(int2str(ix),",",int2str(jx)));
%         final_res(ix,jx) = mean(avg);
%         writematrix(avg,(strcat(int2str(ix),'_',int2str(jx),'.csv')));
%     end
% end
