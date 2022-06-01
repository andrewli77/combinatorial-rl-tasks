import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.hier_base import HierBaseAlgo

class HierPPOAlgo(HierBaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, hi_acmodel, lo_acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 lo_entropy_coef=0.01, hi_entropy_coef=0.1, em_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None,
                 skill_len=100, n_skills = 3
        ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, hi_acmodel, lo_acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, lo_entropy_coef,
                         hi_entropy_coef, em_coef, value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, skill_len, n_skills)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_shape = envs[0].action_space.shape

        assert self.batch_size % self.recurrence == 0

        self.hi_optimizer = torch.optim.Adam(self.hi_acmodel.parameters(), lr, eps=adam_eps)
        self.lo_optimizer = torch.optim.Adam(self.lo_acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        logs_lo = self.update_lo_parameters(exps)
        logs_hi = self.update_hi_parameters(exps)

        logs_combined = {}

        for k,v in logs_lo.items():
            logs_combined['lo_'+k] = v

        for k,v in logs_hi.items():
            logs_combined['hi_'+k] = v

        return logs_combined

    def update_lo_parameters(self, exps):
        lo_exps = exps[0]
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_em_dists = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes_lo():
                # Initialize batch values

                batch_entropy = 0
                batch_em_dist = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory
                sb = lo_exps[inds]

                # Compute loss
                dist, value = self.lo_acmodel(sb.obs)
                entropy = dist.entropy().mean()
                em_dist = self.earth_mover_dist(sb.obs)

                # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                if (len(self.act_shape) == 1): # Not scalar actions (multivariate)
                    delta_log_prob = torch.sum(delta_log_prob, dim=1)
                ratio = torch.exp(delta_log_prob)
                surr1 = ratio * sb.advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                surr1 = (value - sb.returnn).pow(2)
                surr2 = (value_clipped - sb.returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = policy_loss \
                        - self.lo_entropy_coef * entropy \
                        + self.value_loss_coef * value_loss \
                        - self.em_coef * em_dist

                # Update batch values

                batch_entropy += entropy.item()
                batch_em_dist += em_dist.item()
                batch_value += value.mean().item()
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_loss += loss

                # Update actor-critic

                self.lo_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.lo_acmodel.parameters() if p.requires_grad) ** 0.5
                torch.nn.utils.clip_grad_norm_([p for p in self.lo_acmodel.parameters() if p.requires_grad], self.max_grad_norm)
                self.lo_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_em_dists.append(batch_em_dist)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "em_dist": numpy.mean(log_em_dists),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def update_hi_parameters(self, exps):
        hi_exps = exps[1]
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes_hi():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory
                sb = hi_exps[inds]

                # Compute loss
                dist, value = self.hi_acmodel(sb.obs)
                entropy = dist.entropy().mean()

                # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                # if (len(self.act_shape) == 1): # Not scalar actions (multivariate)
                #     delta_log_prob = torch.sum(delta_log_prob, dim=1)
                ratio = torch.exp(delta_log_prob)
                surr1 = ratio * sb.advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                surr1 = (value - sb.returnn).pow(2)
                surr2 = (value_clipped - sb.returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = policy_loss - self.hi_entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update batch values

                batch_entropy += entropy.item()
                batch_value += value.mean().item()
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_loss += loss

                # Update actor-critic

                self.hi_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.hi_acmodel.parameters() if p.requires_grad) ** 0.5
                torch.nn.utils.clip_grad_norm_([p for p in self.hi_acmodel.parameters() if p.requires_grad], self.max_grad_norm)
                self.hi_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes_lo(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames_lo, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def _get_batches_starting_indexes_hi(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames_hi, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc_hi != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def earth_mover_dist(self, obss):
        # Return the mean of all pairwise (squared) 2-Wasserstein distances between the policies conditioned on each skill
        # e.g. W_2( pi(a|s,z_1), pi(a|s,z_2) )^2 ; for all z_1 != z_2.
        # The states s are sampled from the collected data batch. 
        # We do this to encourage the distributions for each skill to be unique.  

        with torch.no_grad():
            hi_dist, _ = self.hi_acmodel(obss)

        batch_size = obss.skill.shape[0]

        def make_skill_vec(obs, z):
            # Set the skill to z
            skills_vec = torch.zeros(obs.skill.shape)
            skills_vec[:, z] = torch.ones(batch_size)
            modified_obs = obs
            modified_obs.skill = skills_vec.to(self.device)
            return modified_obs

        # Make the observation vector conditioned on each skill
        lo_dists = []
        for z in range(self.n_skills):
            lo_dists.append(self.lo_acmodel(make_skill_vec(obss, z))[0])

        w2_dist = 0
        # Compute pairwise Earth mover distances
        for z1 in range(self.n_skills):
            for z2 in range(z1, self.n_skills):
                w2_dist += torch.norm(lo_dists[z1].mean - lo_dists[z2].mean, p=2, dim=1)**2 \
                            + torch.norm(lo_dists[z1].stddev - lo_dists[z2].stddev, p=2, dim=1)**2

        return 2 * w2_dist.mean() / self.n_skills / (self.n_skills - 1)
