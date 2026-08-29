import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Dict, Any, Optional
from CytoBridge.tl.core.interaction import ExpNormalSmearing, cal_interaction

ACTIVATION_FN = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _resolve_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "leakyrelu":
        name = "leaky_relu"
    if name not in ACTIVATION_FN:
        raise ValueError(f"Activation '{name}' not recognized.")
    return ACTIVATION_FN[name]


class SpatialVelocityNet(nn.Module):
    def __init__(
        self,
        in_out_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "leaky_relu",
        use_spatial: bool = False,
        spatial_dim: int = 2,
        residual: bool = False,
    ) -> None:
        super().__init__()
        act_fn = _resolve_activation(activation)
        self.use_spatial = use_spatial
        self.spatial_dim = spatial_dim

        layers = [in_out_dim + 1]
        for _ in range(n_layers):
            layers.append(hidden_dim)
        layers.append(in_out_dim)

        if self.use_spatial:
            self.spatial_net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layers[i], layers[i + 1]),
                        act_fn(),
                    )
                    for i in range(len(layers) - 2)
                ]
            )
            self.spatial_out = nn.Linear(layers[-2], self.spatial_dim)
            self.gene_net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layers[i], layers[i + 1]),
                        act_fn(),
                    )
                    for i in range(len(layers) - 2)
                ]
            )
            self.gene_out = nn.Linear(layers[-2], in_out_dim - self.spatial_dim)
        else:
            self.net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layers[i], layers[i + 1]),
                        act_fn(),
                    )
                    for i in range(len(layers) - 2)
                ]
            )
            self.out = nn.Linear(layers[-2], layers[-1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # DynamicalModel now uses net_input order [t, x].

        if self.use_spatial:
            out = None
            for idx, layer in enumerate(self.spatial_net):
                out = layer(state) if idx == 0 else layer(out)
            spatial_x = self.spatial_out(out)

            out = None
            for idx, layer in enumerate(self.gene_net):
                out = layer(state) if idx == 0 else layer(out)
            gene_x = self.gene_out(out)
            return torch.cat([spatial_x, gene_x], dim=1)

        out = None
        for idx, layer in enumerate(self.net):
            out = layer(state) if idx == 0 else layer(out)
        return self.out(out)


class HyperNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 400,
        n_layers: int = 2,
        activation: str = "leaky_relu",
        residual: bool = False,
    ):
        super().__init__()

        act_fn = _resolve_activation(activation)
        self.n_layers = n_layers
        self.residual = residual

        if self.n_layers == 0:
            self.input_layer = nn.Linear(input_dim, output_dim)
            self.hidden_layers = nn.ModuleList([])
            self.output_layer = nn.Identity()
        else:
            self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), act_fn())

            self.hidden_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act_fn())
                    for _ in range(n_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers == 0:
            return self.input_layer(x)

        x = self.input_layer(x)

        for layer in self.hidden_layers:
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)

        x = self.output_layer(x)
        return x


class InteractionModel(nn.Module):
    def __init__(
        self,
        x_dim,
        n_layers=2,
        hidden_dim=400,
        activation="leaky_relu",
        num_rbf=16,
        cutoff=1,
        dim_reduce=False,
        residual=False,
    ):
        act_fn = _resolve_activation(activation)
        self.n_layers = n_layers
        self.residual = residual
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_expansion = ExpNormalSmearing(
            cutoff=cutoff, num_rbf=self.num_rbf, trainable=False
        )

        if n_layers == 0:
            self.input_layer = nn.Linear(self.num_rbf, 1)
            self.hidden_layers = nn.ModuleList([])
            self.output_layer = nn.Identity()
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(self.num_rbf, hidden_dim), act_fn()
            )
            self.hidden_layers = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act_fn())
                for _ in range(n_layers - 1)
            )
            self.output_layer = nn.Linear(hidden_dim, 1)

        self.cutoff = cutoff
        self.eps = 1e-6
        self.dim_reduce = dim_reduce
        if self.dim_reduce:
            self.pca = nn.Linear(x_dim, 10, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x_t):
        if self.cutoff == 0:
            return 0 * x_t.sum()
        if self.dim_reduce:
            x_t = self.pca(x_t)
        dis = self.compute_distance(x_t)
        dis_exp = self.rbf_expansion(dis[dis != 0])

        if self.n_layers == 0:
            return self.input_layer(dis_exp)
        x = self.input_layer(dis_exp)
        for layer in self.hidden_layers:
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)
        potential = self.output_layer(x)
        return potential

    def compute_distance(self, x):
        return torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps)


class DynamicalModel(nn.Module):
    def __init__(
        self, latent_dim: int, config: Dict[str, Any], use_growth_in_ode_inter=True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.components = list(config["components"])
        if len(set(self.components)) != len(self.components):
            raise ValueError("model.components must not contain duplicate entries.")
        has_interaction_component = "interaction" in self.components
        has_interaction_config = "interaction_net" in config
        if has_interaction_component != has_interaction_config:
            if has_interaction_component:
                raise ValueError(
                    "model.components contains 'interaction', but model.interaction_net "
                    "is missing."
                )
            raise ValueError(
                "model.interaction_net is present while model.components omits "
                "'interaction'; remove the residual interaction configuration."
            )
        self.interaction_type = config.get("interaction_type", "potential")
        self.interaction_group_size = config.get("interaction_group_size", 1024)
        self.net_input_dim = self.latent_dim + 1  # 时间+状态（t + x）
        for comp_name in self.components:
            net_name = f"{comp_name}_net"
            if net_name not in self.config:
                raise ValueError(
                    f"Configuration for component '{comp_name}' not found."
                )

            comp_config = self.config[net_name].copy()

            if comp_name == "velocity":
                use_spatial = comp_config.pop("use_spatial", False)
                if use_spatial:
                    network = SpatialVelocityNet(
                        in_out_dim=self.latent_dim,
                        use_spatial=use_spatial,
                        spatial_dim=config.get("spatial_dim", 2),
                        **comp_config,
                    )
                else:
                    network = HyperNetwork(
                        input_dim=self.net_input_dim,
                        output_dim=self.latent_dim,
                        **comp_config,
                    )

            elif comp_name == "growth":
                network = HyperNetwork(
                    input_dim=self.net_input_dim, output_dim=1, **comp_config
                )

            elif comp_name == "score":
                network = HyperNetwork(
                    input_dim=self.net_input_dim, output_dim=1, **comp_config
                )

            elif comp_name == "interaction":
                self.use_growth_in_ode_inter = use_growth_in_ode_inter
                # Interaction is an optional component in the matched
                # no-interaction ablation.  Construct it in a forked CPU RNG
                # scope so adding/removing it never moves the shared stream
                # used by retained components or subsequent training.
                with torch.random.fork_rng(devices=[]):
                    if self.interaction_type == "gnn":
                        from CytoBridge.tl.graph.spatial_gnn import GNNInteraction

                        network = GNNInteraction(
                            in_out_dim=self.latent_dim,
                            hidden_dim=comp_config.get("hidden_dim", 256),
                            num_heads=comp_config.get("num_heads", 8),
                            num_layers=comp_config.get("num_layers", 1),
                            activation=comp_config.get("activation", "leakyrelu"),
                            num_rbf=comp_config.get("num_rbf", 8),
                            cutoff=comp_config.get("cutoff", 0.2),
                            use_spatial=comp_config.get("use_spatial", True),
                            spatial_dim=config.get("spatial_dim", 2),
                            rbf_trainable=comp_config.get("rbf_trainable", False),
                            edge_predictor_path=comp_config.get("edge_predictor_path"),
                            edge_predictor_thre=comp_config.get(
                                "edge_predictor_thre", 0.5
                            ),
                            edge_predictor_root=comp_config.get("edge_predictor_root"),
                            edge_prior_mode=comp_config.get(
                                "edge_prior_mode", "learned"
                            ),
                            load_edge_predictor_from_path=comp_config.get(
                                "load_edge_predictor_from_path", True
                            ),
                        )
                    else:
                        network = InteractionModel(self.latent_dim, **comp_config)
            else:
                raise ValueError(f"Unknown dynamical component: '{comp_name}'")

            self.add_module(f"{comp_name}_net", network)

    def _expand_time_like(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand/reshape t to [N, 1] aligned with x rows."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(device=x.device, dtype=x.dtype)

        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.dim() != 2 or t.shape[1] != 1:
            raise ValueError(f"Expected t shape [N] or [N,1], got {tuple(t.shape)}")

        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0], 1)
        elif t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Time rows ({t.shape[0]}) must match x rows ({x.shape[0]})"
            )
        return t

    def build_net_input(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Build network input with canonical order [t, x]."""
        t_expanded = self._expand_time_like(t, x)
        return torch.cat([t_expanded, x], dim=1)

    def predict_velocity(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if "velocity" not in self.components:
            raise ValueError("Model does not contain 'velocity' component.")
        return self.velocity_net(self.build_net_input(t, x))

    def predict_growth(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if "growth" not in self.components:
            raise ValueError("Model does not contain 'growth' component.")
        return self.growth_net(self.build_net_input(t, x))

    def predict_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if "score" not in self.components:
            raise ValueError("Model does not contain 'score' component.")
        return self.score_net(self.build_net_input(t, x))

    def predict_components(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        lnw: Optional[torch.Tensor] = None,
        include_interaction: bool = True,
        include_score_gradient: bool = True,
        score_create_graph: bool = True,
        interaction_generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified component prediction entrypoint used by training and downstream."""
        outputs: Dict[str, torch.Tensor] = {}

        if "velocity" in self.components:
            outputs["velocity"] = self.predict_velocity(t, x)

        if "growth" in self.components:
            outputs["growth"] = self.predict_growth(t, x)

        if "score" in self.components:
            x_for_score = x.requires_grad_(True)
            out_score, gradient = self.compute_score(
                t=t,
                x=x_for_score,
                create_graph=score_create_graph,
            )
            outputs["score"] = out_score
            if include_score_gradient:
                outputs["score_gradient"] = gradient

        if include_interaction and "interaction" in self.components:
            if lnw is None:
                raise ValueError("lnw is required when include_interaction=True")
            if self.interaction_type == "gnn":
                outputs["interaction"] = cal_interaction(
                    x,
                    lnw,
                    self.interaction_net,
                    m=self.interaction_group_size,
                    use_mass=self.use_growth_in_ode_inter,
                    t=t,
                    generator=interaction_generator,
                ).float()
            else:
                outputs["interaction"] = cal_interaction(
                    x,
                    lnw,
                    self.interaction_net,
                    cutoff=self.interaction_net.cutoff,
                    use_mass=self.use_growth_in_ode_inter,
                    generator=interaction_generator,
                ).float()
        return outputs

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        lnw: torch.Tensor,
        except_interaction: bool = True,
        interaction_generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.predict_components(
            t=t,
            x=x,
            lnw=lnw,
            include_interaction=except_interaction,
            include_score_gradient=True,
            score_create_graph=True,
            interaction_generator=interaction_generator,
        )

    def compute_score(self, t, x, create_graph: bool = True):
        x = x.requires_grad_(True)
        out_score = self.predict_score(t=t, x=x)
        gradient = torch.autograd.grad(
            outputs=out_score,
            inputs=x,
            grad_outputs=torch.ones_like(out_score),
            create_graph=create_graph,
        )[0]
        return out_score, gradient
