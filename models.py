import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from utils import *
from utils import homo_warp
from inplace_abn import InPlaceABN
from renderer import run_network_mvs


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        self.freq_bands = freq_bands.reshape(1,-1,1).cuda()

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        repeat = inputs.dim()-1
        inputs_scaled = (inputs.unsqueeze(-2) * self.freq_bands.view(*[1]*repeat,-1,1)).reshape(*inputs.shape[:-1],-1)
        inputs_scaled = torch.cat((inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)),dim=-1)
        return inputs_scaled

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn

class Renderer_v0(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        """
        super(Renderer_v0, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) + bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class Renderer_ours(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        """
        super(Renderer_ours, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in range(D-1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class Renderer_color_fusion(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4],use_viewdirs=False):
        """
        """
        super(Renderer_color_fusion, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=True)] + [
                nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)

        attension_dim = 16 + 3 + self.in_ch_views//3 #  16 + rgb dim + angle dim
        self.ray_attention = MultiHeadAttention(4, attension_dim, 4, 4)

        if use_viewdirs:
            self.feature_linear = nn.Sequential(nn.Linear(W, 16), nn.ReLU())
            self.alpha_linear = nn.Sequential(nn.Linear(W, 1), nn.ReLU())
            self.rgb_out = nn.Sequential(nn.Linear(attension_dim, 3),nn.Sigmoid())  #
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_out.apply(weights_init)

    def forward_alpha(self,x):
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, self.in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha


    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim - self.in_ch_pts - self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)

        # color
        input_views = input_views.reshape(-1, 3, self.in_ch_views//3)
        rgb = input_feats[..., 8:].reshape(-1, 3, 4)
        rgb_in = rgb[..., :3]

        N = rgb.shape[0]
        feature = self.feature_linear(h)
        h = feature.reshape(N, 1, -1).expand(-1, 3, -1)
        h = torch.cat((h, input_views, rgb_in), dim=-1)
        h, _ = self.ray_attention(h, h, h, mask=rgb[...,-1:])
        rgb = self.rgb_out(h)

        rgb = torch.sum(rgb , dim=1).reshape(*alpha.shape[:2], 3)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs




class MVSNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch_pts=3, input_ch_views=3, input_ch_feat=8, skips=[4], net_type='v2'):
        """
        """
        super(MVSNeRF, self).__init__()

        self.in_ch_pts, self.in_ch_views,self.in_ch_feat = input_ch_pts, input_ch_views, input_ch_feat

        # we provide two version network structure
        if 'v0' == net_type:
            self.nerf = Renderer_ours(D=D, W=W,input_ch_feat=input_ch_feat,
                     input_ch=input_ch_pts, output_ch=4, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)
        elif 'v1' == net_type:
            self.nerf = Renderer_color_fusion(D=D, W=W,input_ch_feat=input_ch_feat,
                     input_ch=input_ch_pts, output_ch=4, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)



    def forward(self, x):
        RGBA = self.nerf(x)
        return RGBA

def create_nerf_mvs(args, pts_embedder=True, use_mvs=False, dir_embedder=True):
    """Instantiate mvs NeRF's MLP model.
    """

    if pts_embedder:
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed, input_dims=args.pts_dim)
    else:
        embed_fn, input_ch = None, args.pts_dim

    embeddirs_fn = None
    if dir_embedder:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, input_dims=args.dir_dim)
    else:
        embeddirs_fn, input_ch_views = None, args.dir_dim


    skips = [4]
    model = MVSNeRF(D=args.netdepth, W=args.netwidth,
                 input_ch_pts=input_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_feat=args.feat_dim, net_type=args.net_type).to(device)

    grad_vars = []
    grad_vars += list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = MVSNeRF(D=args.netdepth, W=args.netwidth,
                 input_ch_pts=input_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_feat=args.feat_dim).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda pts, viewdirs, rays_feats, network_fn: run_network_mvs(pts, viewdirs, rays_feats, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    EncodingNet = None
    if use_mvs:
        EncodingNet = MVSNet().to(device)
        grad_vars += list(EncodingNet.parameters())    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    start = 0


    ##########################

    # Load checkpoints
    ckpts = []
    if args.ckpt is not None and args.ckpt != 'None':
        ckpts = [args.ckpt]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 :
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load model
        if use_mvs:
            state_dict = ckpt['network_mvs_state_dict']
            EncodingNet.load_state_dict(state_dict)

        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'network_mvs': EncodingNet,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }


    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

###################################  feature net  ######################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))


    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x

class MVSNet(nn.Module):
    def __init__(self,
                 num_groups=1,
                 norm_act=InPlaceABN,
                 levels=1):
        super(MVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [128,32,8]
        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = FeatureNet()

        self.N_importance = 0
        self.chunk = 1024

        self.cost_reg_2 = CostRegNet(32 + 9, norm_act)


    def build_volume_costvar(self, feats, proj_mats, depth_values, pad=0, idx=[0,1,2]):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]


        D = depth_values.shape[1]
        ref_feats, src_feats  = feats[:, idx[0]], feats[:, idx[1:]]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        if proj_mats.dim() == 5:
            proj_mats = (proj_mats[0,0,idx[1:]] @ proj_mats[1,0,idx[0]])[None,:,:3]
        else:
            proj_mats = proj_mats[:, idx[1:]]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)


        if pad>0:
            ref_feats = F.pad(ref_feats, (pad,pad,pad,pad), "constant", 0)


        source_volume = []
        source_volume.append(ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1))  # (B, C, D, h, w))
        for src_feat, proj_mat in zip(src_feats, proj_mats):
            warped_volume, src_grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            source_volume.append(warped_volume)


        cost_var = torch.cat(source_volume, dim=0)
        del source_volume
        cost_var = torch.cat((cost_var, torch.mean(cost_var**2, dim=0, keepdim=True)-torch.mean(cost_var, dim=0, keepdim=True)**2),dim=0)
        cost_var = cost_var.view(1,-1,*cost_var.shape[2:])
        return cost_var


    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0, idx=[0,1,2]):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, idx[0]], feats[:, idx[1:]]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, idx[1:]]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)


        if pad>0:
            ref_feats = F.pad(ref_feats, (pad,pad,pad,pad), "constant", 0)

        img_feat = torch.empty((B, 9 +32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B*V,*imgs.shape[2:]),(H,W),mode='bilinear',align_corners=False).view(B,V,-1,H,W).permute(1,0,2,3,4)
        img_feat[:,:3,:,pad:H+pad,pad:W+pad] = imgs[idx[0]].unsqueeze(2).expand(-1,-1,D,-1,-1)


        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        count = torch.ones((B, 1, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i,(src_img, src_feat, proj_mat) in enumerate(zip(imgs[idx[1:]], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:,(i+1)*3:(i+2)*3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid >-1.0)*(grid < 1.0))
            in_mask = (in_mask[...,0]*in_mask[...,1])
            count += in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / count
        img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        return img_feat


    def forward(self, imgs, proj_mats, near_far, pad=0, idx=[0,1,2]):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)

        B, V, _, H, W = imgs.shape

        imgs = imgs.reshape(B * V, 3, H, W)
        feats = self.feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)

        imgs = imgs.view(B, V, 3, H, W)

        l = 0
        level = 2 - l
        feats_l = feats  # (B*V, C, h, w)

        feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)


        D = 128
        init_depth_min = near_far[0]  # assume batch size==1
        depth_interval_l = near_far[1] - near_far[0]
        depth_values = init_depth_min + \
                       depth_interval_l * \
                       torch.linspace(0., 1., steps=D,
                                    device=imgs.device,
                                    dtype=imgs.dtype) # (B, D)
        depth_values = depth_values.unsqueeze(0)

        volume_feat = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad, idx=idx)
        # volume_feat = self.build_volume_costvar(feats_l, proj_mats, depth_values, pad=pad, idx=idx)
        volume_feat = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])

        return volume_feat, feats_l, depth_values


class RefVolume(nn.Module):
    def __init__(self, volume):
        super(RefVolume, self).__init__()

        self.feat_volume = nn.Parameter(volume)

    def forward(self, ray_coordinate_ref):
        '''coordinate: [N, 3]
            z,x,y
        '''

        device = self.feat_volume.device
        H, W = ray_coordinate_ref.shape[-3:-1]
        grid = ray_coordinate_ref.view(-1, 1, H, W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
        features = F.grid_sample(self.feat_volume, grid, align_corners=True, mode='bilinear')[:, :, 0].permute(2, 3, 0,1).squeeze()
        return features
