# EC2 Instance Setup

## Docker installation
- sudo yum update -y
- sudo amazon-linux-extras install docker
- sudo service docker start
- sudo usermod -a -G docker ec2-user
- Logout and login
- `docker info`

## Bento Server
- In Security group, change inbound rule protocol to TCP
- `docker pull ghcr.io/samhita-alla/bento:0.0.1`
- `docker run -p 80:3000 ghcr.io/samhita-alla/bento:0.0.1`

## Test API
- `python services/bentoml/test_api.py --url http://13.59.21.71/`
