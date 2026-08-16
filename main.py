from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
import copy
from anchor_selection import compute_anchor_budget

from args import *
from model import *
from utils import *
from dataset import *
def os_arg():
    if not os.path.isdir('results'):
        os.mkdir('results')
    # args
    args = make_args()
    # print(args)
    #np.random.seed(123)
    #np.random.seed()
    writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
    writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
    writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')

    return args,writer_train,writer_val,writer_test

if __name__ == '__main__':
    args,writer_train,writer_val,writer_test=os_arg()
    print(args)
    # set up gpu
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print('Using CPU')
    device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')


    if args.task == 'both':
        tasks = ['link', 'link_pair']
    else:
        tasks = [args.task]

    for task in tasks:
        args.task = task  # مقدار آرگومان رو آپدیت کن که همه جا درست کار کنه
        if args.dataset=='All':
            if task == 'link':
                datasets_name = ['grid','communities','ppi', 'Amazon-Computers','Amazon-Photo','Flickr',
                                 'PubMed', 'karate', 'Wiki','BlogCatalog','Facebook','Minesweeper','Tolokers']
            else:
                datasets_name = ['communities', 'email', 'protein', 'Amazon-Computers','Amazon-Photo','Flickr',
                                 'PubMed', 'karate', 'Wiki','BlogCatalog','Facebook','Minesweeper','Tolokers']
        else:
            datasets_name = [args.dataset]
        print(f"\n🔁 Now running TASK: {task.upper()} on dataset(s): {datasets_name}\n")

        for dataset_name in datasets_name:

            results = []
        
            models_with_agg = ['PGNN', 'ATTSP']
        
            # ============================================
            # 5 independent runs
            # ============================================
            for repeat in range(args.repeat_num):
        
                best_val_auc = -1.0
                best_epoch = -1
                best_model_state = None
        
                # ----------------------------------------
                # Seed of this run
                # repeat 0 -> 123
                # repeat 1 -> 124
                # ...
                # ----------------------------------------
                repeat_seed = args.base_seed + repeat
                set_seed(repeat_seed)
        
                print(
                    f"\n========================================"
                )
                print(
                    f"Repeat {repeat} | seed = {repeat_seed}"
                )
                print(
                    f"========================================\n"
                )
        
                # ========================================
                # NEW DATA + NEW SPLIT FOR THIS RUN
                # ========================================
                time1 = time.time()
        
                data_list = get_tg_dataset(
                    args,
                    dataset_name,
                    use_cache=False,
                    remove_feature=args.rm_feature
                )
        
                time2 = time.time()
        
                print(
                    dataset_name,
                    'load time',
                    time2 - time1
                )
        
                # ========================================
                # Dataset information
                # ========================================
                input_dim = data_list[0].x.shape[1]
                output_dim = args.output_dim
        
                num_features = input_dim
                num_node_classes = None
                num_graph_classes = None
        
                if (
                    'y' in data_list[0].__dict__
                    and data_list[0].y is not None
                ):
                    num_node_classes = max(
                        [data.y.max().item() for data in data_list]
                    ) + 1
        
                if (
                    'y_graph' in data_list[0].__dict__
                    and data_list[0].y_graph is not None
                ):
                    num_graph_classes = max(
                        [data.y_graph.numpy()[0] for data in data_list]
                    ) + 1
        
                print(
                    'Dataset',
                    dataset_name,
                    'Graph',
                    len(data_list),
                    'Feature',
                    num_features,
                    'Node Class',
                    num_node_classes,
                    'Graph Class',
                    num_graph_classes
                )
        
                nodes = [
                    data.num_nodes
                    for data in data_list
                ]
        
                edges = [
                    data.num_edges
                    for data in data_list
                ]
        
                print(
                    'Node: max{}, min{}, mean{}'.format(
                        max(nodes),
                        min(nodes),
                        sum(nodes) / len(nodes)
                    )
                )
        
                print(
                    'Edge: max{}, min{}, mean{}'.format(
                        max(edges),
                        min(edges),
                        sum(edges) / len(edges)
                    )
                )
        
                args.batch_size = min(
                    args.batch_size,
                    len(data_list)
                )
        
                print(
                    'Anchor num {}, Batch size {}'.format(
                        args.anchor_num,
                        args.batch_size
                    )
                )
        
                # ========================================
                # Anchor budget
                # ========================================
                if args.anchor_method in [
                    'slot_joint',
                    'global_topk'
                ]:
        
                    K_per_graph = [
                        compute_anchor_budget(
                            num_nodes=data.num_nodes,
                            mode=args.anchor_budget,
                            fixed_k=args.anchor_num,
                            reduction=args.anchor_reduction,
                            exact_fixed=args.anchor_fixed_exact,
                        )
                        for data in data_list
                    ]
        
                    if args.anchor_method == 'slot_joint':
                        slot_k_max = max(K_per_graph)
                    else:
                        slot_k_max = None
        
                    print(
                        f"{args.anchor_method} anchor budget:",
                        f"K_min={min(K_per_graph)},",
                        f"K_max={max(K_per_graph)}"
                    )
        
                else:
                    slot_k_max = None
        
                # ========================================
                # Anchor preselection
                # ========================================
                for i, data in enumerate(data_list):
        
                    if args.anchor_method not in [
                        'slot_joint',
                        'global_topk'
                    ]:
        
                        preselect_anchor(
                            data,
                            layer_num=args.layer_num,
                            anchor_num=args.anchor_num,
                            device='cpu',
                            args=args
                        )
        
                    data = data.to(device)
                    data_list[i] = data
        
                # ========================================
                # From HERE continue with your current
                # Guard + model creation + optimizer
                # + training loop
                # ========================================

                # =====================================================
                # Guard: new learnable anchor methods are PGNN-only
                # =====================================================
                if (
                    args.anchor_method in ['slot_joint', 'global_topk']
                    and args.model != 'PGNN'
                ):
                    raise ValueError(
                        f"anchor_method='{args.anchor_method}' "
                        "is currently implemented for model='PGNN' only."
                    )
        
                # ==== ساخت مدل ====
                if args.model in models_with_agg:
                    model = locals()[args.model](
                        input_dim=input_dim,
                        feature_dim=args.feature_dim,
                        hidden_dim=args.hidden_dim,
                        output_dim=output_dim,
                        feature_pre=args.feature_pre,
                        layer_num=args.layer_num,
                        dropout=args.dropout,
                        aggregation=args.aggregation,
                        comb_mode=args.comb_mode,
                        prob_factor=getattr(args, 'prob_factor', 5),
                        prob_min_top=getattr(args, 'prob_min_top', 1),
                        prob_context_mode=getattr(args, 'prob_context_mode', 'concat'),

                        anchor_method=args.anchor_method,
                        anchor_budget=args.anchor_budget,
                        anchor_num=args.anchor_num,
                        anchor_reduction=args.anchor_reduction,
                        anchor_fixed_exact=args.anchor_fixed_exact,
                        
                        slot_k_max=slot_k_max,
                        
                        selector_dim=args.hidden_dim,
                        slot_temperature=1.0,
                        sinkhorn_iters=30,
                    ).to(device)
                else:
                    model = locals()[args.model](
                        input_dim=input_dim,
                        feature_dim=args.feature_dim,
                        hidden_dim=args.hidden_dim,
                        output_dim=output_dim,
                        feature_pre=args.feature_pre,
                        layer_num=args.layer_num,
                        dropout=args.dropout,
                        comb_mode=args.comb_mode,
                        prob_factor=getattr(args, 'prob_factor', 5),
                        prob_min_top=getattr(args, 'prob_min_top', 1),
                        prob_context_mode=getattr(args, 'prob_context_mode', 'concat'),
                    ).to(device)
        
                
                # loss
                params = list(model.parameters())
                
                # اگر روش learnable فعال است، پارامترهای MLPها را هم اضافه کن
                if args.anchor_method == 'learnable_hybrid':
                    for data in data_list:
                        if hasattr(data, 'anchor_selector'):
                            params += list(data.anchor_selector.parameters())
                optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
                if 'link' in args.task:
                    loss_func = nn.BCEWithLogitsLoss()
                    out_act = nn.Sigmoid()


                for epoch in range(args.epoch_num):
                    if epoch==200:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] /= 10
                    model.train()
                    optimizer.zero_grad()
                    shuffle(data_list)
                    effective_len = len(data_list)//args.batch_size*len(data_list)
                    for id, data in enumerate(data_list[:effective_len]):
                        # --- انتخاب انکرها ---
                        # حالت‌های معمولی (random, degree, hyper, ...)
                        if (
                            args.permute
                            and args.anchor_method not in [
                                'learnable_hybrid',
                                'slot_joint',
                                'global_topk'
                            ]
                        ):
                            preselect_anchor(
                                data,
                                layer_num=args.layer_num,
                                anchor_num=args.anchor_num,
                                device=device,
                                args=args
                            )

                        # حالت learnable_hybrid: refresh سبک مخصوص خودش
                        if args.anchor_method == 'learnable_hybrid':
                            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device, args=args)
                        # print(data)
                        out = model(data)
                        # get_link_mask(data,resplit=False)  # resample negative links
                        edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
                        label = torch.cat((label_positive,label_negative)).to(device)
                        loss = loss_func(pred, label)
                        # === Neighbor-sim (cos) روی یال‌های TRAIN مثبت ===
                        if getattr(args, 'lambda_ns', 0.0) > 0:
                            loss_ns = neighbor_sim_loss(out, data.mask_link_positive_train, device, mode=args.ns_mode)
                            loss = loss + args.lambda_ns * loss_ns

                        # === Auxiliary Loss برای learnable_hybrid ===
                        if args.anchor_method == 'learnable_hybrid' and hasattr(data, 'anchor_softmask'):
                            # اطمینان از وجود anchor_scores
                            if hasattr(data, 'anchor_scores'):
                                probs = torch.softmax(data.anchor_scores, dim=0)
                                entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum()
                            else:
                                entropy = 0.0

                            # تنوع در انتخاب انکرها (soft mask)
                            diversity = 1.0 - (data.anchor_softmask @ data.anchor_softmask)

                            # ترکیب دو بخش entropy و diversity
                            loss_aux = args.anchor_aux_w * (0.5 * entropy + 0.5 * diversity)

                            # افزودن به loss اصلی
                            loss = loss + loss_aux


                        """
                        ### update old for multiple graphs ###
                        loss.backward()
                        if id % args.batch_size == args.batch_size-1:
                            if args.batch_size>1:
                                # if this is slow, no need to do this normalization
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad /= args.batch_size
                            optimizer.step()
                            optimizer.zero_grad()
                        """
                        ### update new for multiple graphs ###
                        loss.backward()
                        
                        is_batch_end = ((id + 1) % args.batch_size == 0)
                        is_last_graph = (id == len(data_list) - 1)
                        
                        if is_batch_end or is_last_graph:
                        
                            if is_batch_end:
                                current_batch_size = args.batch_size
                            else:
                                # تعداد گراف‌های باقی‌مانده در batch آخر
                                current_batch_size = (id + 1) % args.batch_size
                        
                            if current_batch_size > 1:
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad /= current_batch_size
                        
                            optimizer.step()
                            optimizer.zero_grad()

                    

                    if epoch % args.epoch_log == 0:
                        # evaluate
                        model.eval()
                        loss_train = 0
                        loss_val = 0
                        correct_train = 0
                        all_train = 0
                        correct_val = 0
                        all_val = 0
                        auc_train = 0
                        auc_val = 0
                        emb_norm_min = 0
                        emb_norm_max = 0
                        emb_norm_mean = 0
                        for id, data in enumerate(data_list):
                            out = model(data)
                            emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                            emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                            emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

                            # train
                            # get_link_mask(data, resplit=False)  # resample negative links
                            edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                            pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                            label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                            label = torch.cat((label_positive, label_negative)).to(device)
                            loss_train += loss_func(pred, label).cpu().data.numpy()
                            auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                            # val
                            edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                            pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                            label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                            label = torch.cat((label_positive, label_negative)).to(device)
                            loss_val += loss_func(pred, label).cpu().data.numpy()
                            auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

                        loss_train /= id+1
                        loss_val /= id+1
                        emb_norm_min /= id+1
                        emb_norm_max /= id+1
                        emb_norm_mean /= id+1
                        auc_train /= id+1
                        auc_val /= id+1

                        if auc_val > best_val_auc:
                            best_val_auc = auc_val
                            best_epoch = epoch
                            best_model_state = copy.deepcopy(model.state_dict())

                        print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                            'Val AUC: {:.4f}'.format(auc_val))
                        writer_train.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
                        writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
                        writer_val.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
                        writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)


                # ========================================
                # Final test: evaluate exactly once
                # using the checkpoint with best validation AUC
                # ========================================
                if best_model_state is None:
                    raise RuntimeError("No validation checkpoint was saved.")

                model.load_state_dict(best_model_state)
                model.eval()

                auc_test = 0.0
                loss_test = 0.0

                with torch.no_grad():
                    for id, data in enumerate(data_list):
                        out = model(data)

                        edge_mask_test = np.concatenate(
                            (data.mask_link_positive_test, data.mask_link_negative_test),
                            axis=-1
                        )
                        nodes_first = torch.index_select(
                            out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device)
                        )
                        nodes_second = torch.index_select(
                            out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device)
                        )
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)

                        label_positive = torch.ones(
                            [data.mask_link_positive_test.shape[1]],
                            dtype=pred.dtype
                        )
                        label_negative = torch.zeros(
                            [data.mask_link_negative_test.shape[1]],
                            dtype=pred.dtype
                        )
                        label = torch.cat((label_positive, label_negative)).to(device)

                        loss_test += loss_func(pred, label).item()
                        auc_test += roc_auc_score(
                            label.detach().cpu().numpy(),
                            out_act(pred).detach().cpu().numpy()
                        )

                loss_test /= len(data_list)
                auc_test /= len(data_list)

                print(
                    f"Best Val AUC = {best_val_auc:.6f} "
                    f"at epoch {best_epoch} | "
                    f"Final Test AUC = {auc_test:.6f}"
                )

                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/auc_' + dataset_name,
                    auc_test,
                    best_epoch
                )
                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/loss_' + dataset_name,
                    loss_test,
                    best_epoch
                )

                results.append(auc_test)
            results = np.array(results)
            results_mean = np.mean(results).round(6)
            results_std = np.std(results).round(6)
            print('-----------------Final-------------------')
            print(results_mean, results_std)
            budget_tag = args.anchor_budget

            if args.anchor_budget == 'progressive':
                budget_tag += f"_r{args.anchor_reduction}"
            
            elif args.anchor_budget == 'fixed':
                budget_tag += f"_k{args.anchor_num}"
            
            
            result_file = (
                f"results/"
                f"{args.task}_"
                f"{args.model}_"
                f"{dataset_name}_"
                f"{args.anchor_method}_"
                f"{budget_tag}_"
                f"layer{args.layer_num}_"
                f"approximate{args.approximate}.txt"
            )
            
            with open(result_file, 'w') as f:
                f.write(f"runs={results.tolist()}\n")
                f.write(f"mean={results_mean}\n")
                f.write(f"std={results_std}\n")

    # export scalar data to JSON for external processing
    writer_train.export_scalars_to_json("./all_scalars.json")
    writer_train.close()
    writer_val.export_scalars_to_json("./all_scalars.json")
    writer_val.close()
    writer_test.export_scalars_to_json("./all_scalars.json")
    writer_test.close()
