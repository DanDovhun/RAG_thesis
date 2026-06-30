##
# Returns list of found businesses:
# $ businesses = Apps::BusinessRegisterApp::BusinessActs.new.search_business('my name')
#
# Returns list of acts for given business
# acts = Apps::BusinessRegisterApp::BusinessActs.new.search_acts(businesses.first)
#
module Apps
  module OrSrApp
    module ActsSubmission
      class BusinessActs
      def search_business(query)
        return [] unless query.present?
        res = client.get('https://orsr.sk/orsr.webapiforms/search', s: query)

        if res.success?
          json = aspx_to_json(res.body)
          Business.array_from_json(json)
        else
          Rails.logger.error res.body
          raise "Api fail"
        end
      end

      def search_acts(business)
        if !business || business.oddiel.blank? || business.vlozka.blank? || business.name.blank?
          return []
        end

        res = client.get(
          'https://sluzby.orsr.sk/lookup/documents',
          section: business.oddiel,
          insertNumber: business.vlozka,
          courtCode: business.name.last(2).first,
        )
        if res.success?
          Act.array_from_json(res.body)
        else
          Rails.logger.error res.body
          raise "Api fail"
        end
      end

      def perform
      end

      def client
        Faraday.new do |faraday|
          faraday.request :json
          faraday.response :json
          faraday.response(:logger, Rails.logger, headers: false, bodies: false)
        end
      end

      def aspx_to_json(str)
        aspx_to_json = str.gsub(/^\(/, '').gsub(/\);$/, '')
        JSON.parse(aspx_to_json.force_encoding('UTF-8'))
      end

      class Business
        include ActiveModel::Model
        attr_accessor(:ico, :name, :address, :oddiel, :vlozka, :sud, :raw)

        def self.array_from_json(json)
          json['aaData'].map do |business|
            sub_data = JSON.parse(business[3])
            new(
              ico: business[0],
              name: business[1],
              address: business[2],
              oddiel: sub_data['oddiel'],
              vlozka: sub_data['vlozka'],
              sud: sub_data['sud'],
              raw: business,
            )
          end
        end
      end

      class Act
        include ActiveModel::Model
        attr_accessor(:raw, :name, :formatted_name, :type, :delivery_date, :serial_number, :page_count, :make_copy, :json_value)

        def self.array_from_json(json)
          json.map do |act|
            new(
              name: act['nazovListiny'],
              formatted_name: act['formattedName'],
              type: act['type'],
              delivery_date: act['datumDorucenia'],
              serial_number: act['serialNumber'],
              page_count: act['pageCount'],
              raw: act,
              make_copy: false,
              json_value: {
                "id" => act['serialNumber'],
                "name" => act['nazovListiny'],
                "code" => act['serialNumber'],
                "make_copy" => true
              }.to_json
            )
          end
        end
      end
      end
    end
  end
end
