container_name='af3_train'
docker run -it --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  --name ${container_name} \
  -v /mnt/nvme0/datasets/af3:/datasets \
  -v /mnt/nvme1:/weights \
  -v /root:/root \
  -w /root/sources/AlphaFold3 \
  vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1

