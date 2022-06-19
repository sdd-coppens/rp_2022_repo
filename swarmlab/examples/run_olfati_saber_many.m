for i = 3 : 20
%     avg = zeros(1,10);
%     for k = 1 : 10
    avg = example_olfati_saber(1, 1, i);
%     end
    disp(int2str(i));
    disp(avg);
    writematrix(avg,(strcat(int2str(i),'.csv')));
end



% final_res = zeros(1, 20, 20);
% avg = zeros(1,20);
% % for i = 0.0 : 0.05 : 1
%     i = 0.45;
%     ix = floor(i*20) + 1;
%     for j = 0.0 : 0.05 : 1
%         avg = zeros(1,20);
%  
%         for k = 1 : 20
% %         %avg = [avg, example_vasarhelyi(i)];
%             avg(k) = example_olfati_saber(i, j);
%         end
%         jx = floor(j*20) + 1;
%         disp(strcat(int2str(ix),",",int2str(jx)));
%         final_res(ix,jx) = mean(avg);
%         writematrix(avg,(strcat(int2str(ix),'_',int2str(jx),'.csv')));
%     end
% % end
% disp(final_res);








% plot(0 : 0.05 : 1, final_res);
% disp(example_vasarhelyi(1, 0.23));



% final_res = zeros(1,20);
% avg = zeros(1,20);
% %for i = 0 : 0.05 : 1
% for i = 0 : 0.05 : 1
%     avg = zeros(1,20);
% 
%     for j = 1 : 20
%         %avg = [avg, example_vasarhelyi(i)];
%         avg(j) = example_olfati_saber(i);
%     end
% 
%     ix = floor(i*20) + 1;
%     disp(ix);
%     final_res(ix) = mean(avg);
%     writematrix(avg,(strcat(int2str(ix),'_olfati_saber.csv')));
% end
% disp(final_res);
% plot(0 : 0.05 : 1, final_res);