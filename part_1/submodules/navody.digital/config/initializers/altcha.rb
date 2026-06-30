# frozen_string_literal: true

Altcha.setup do |config|
  config.algorithm = 'SHA-256'
  config.num_range = (30_000..45_000)
  config.timeout = 5.minutes
  config.hmac_key = Rails.application.config_for(:auth).dig(:altcha, :hmac_key)
end
