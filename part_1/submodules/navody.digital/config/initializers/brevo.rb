require 'brevo'

Brevo.configure do |config|
  config.api_key['api-key'] = Rails.application.config_for(:auth).dig(:brevo, :api_key)
end